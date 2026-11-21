import os
import io
import gc
import time
import ctypes
import tempfile
import threading
import traceback
import warnings
import urllib.request
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm

import eccodes
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.units import units
from metpy.interpolate import cross_section
from pyproj import Geod

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RRFS 3km Cross-Section API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD"],
    allow_headers=["*"],
)

S3_BUCKET_URL = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
DATA_DIR = "./Data"
CACHE_DIR = "./Data/cache"
OROG_PATH = "./Data/rrfs_orog.nc"
CACHE_MAX_AGE_HOURS = 24
MAX_CACHE_FILES = 10

# Every 50 mb from 1000 mb up to 250 mb (16 levels)
TARGET_LEVELS_MB = set(range(250, 1001, 50))

# Prioritize full prslev first because it includes RWMR, SNMR, GRLE, and 1000m REFD
S3_TEMPLATES = [
    f"{S3_BUCKET_URL}/rrfs.{{ymd}}/{{hh}}/rrfs.t{{hh}}z.prslev.3km.f{{fhour}}.conus.grib2",
    f"{S3_BUCKET_URL}/rrfs.{{ymd}}/{{hh}}/rrfs.t{{hh}}z.prslev.3km.f{{fhour}}.na.grib2",
    f"{S3_BUCKET_URL}/rrfs.{{ymd}}/{{hh}}/rrfs.t{{hh}}z.prslevnomads.3km.f{{fhour}}.conus.grib2",
]
PREFERRED_TEMPLATE = None

# Keep at most 2 small cropped datasets in RAM to stay safely under 512 MB container limit
RAM_DATASET_CACHE = OrderedDict()
MAX_RAM_ITEMS = 2
CACHED_OROG_DA = None

# Concurrency controls for multi-user traffic on a 512 MB container
RENDER_SEMAPHORE = threading.Semaphore(2)
_FETCH_LOCKS = {}
_FETCH_LOCKS_GUARD = threading.Lock()
_OROG_LOCK = threading.Lock()
_RAM_CACHE_LOCK = threading.Lock()

# NWS-style reflectivity colormap (5 to 70 dBZ in 5 dBZ steps)
DBZ_LEVELS = np.arange(5, 75, 5)
DBZ_CMAP = ListedColormap([
    "#04e9e7",  # 5-10 dBZ (light cyan)
    "#019ff4",  # 10-15 dBZ (medium blue)
    "#0300f4",  # 15-20 dBZ (dark blue)
    "#02fd02",  # 20-25 dBZ (light green)
    "#01c501",  # 25-30 dBZ (medium green)
    "#008e00",  # 30-35 dBZ (dark green)
    "#fdf802",  # 35-40 dBZ (yellow)
    "#e5bc00",  # 40-45 dBZ (yellow-orange)
    "#fd9500",  # 45-50 dBZ (orange)
    "#fd0000",  # 50-55 dBZ (red)
    "#d40000",  # 55-60 dBZ (dark red)
    "#bc0000",  # 60-65 dBZ (crimson)
    "#f800fd",  # 65-70 dBZ (magenta)
])
DBZ_NORM = BoundaryNorm(DBZ_LEVELS, ncolors=DBZ_CMAP.N, clip=True)


def get_fetch_lock(key: str) -> threading.Lock:
    """Return a per-bbox/cycle lock so simultaneous requests for the same slice don't duplicate S3 work."""
    with _FETCH_LOCKS_GUARD:
        if key not in _FETCH_LOCKS:
            _FETCH_LOCKS[key] = threading.Lock()
        return _FETCH_LOCKS[key]


def release_memory():
    """Force Python garbage collection and ask glibc to return freed heap pages to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def prune_cache(cache_dir=CACHE_DIR, max_age_hours=CACHE_MAX_AGE_HOURS, max_files=MAX_CACHE_FILES):
    """Remove expired cache files and enforce an LRU cap of `max_files` on disk."""
    if not os.path.isdir(cache_dir):
        return

    cutoff = time.time() - (max_age_hours * 3600)
    cached_entries = []

    for entry in os.scandir(cache_dir):
        if not entry.is_file():
            continue
        if entry.name.endswith(".tmp") or entry.name.endswith(".tmp.nc"):
            try:
                if entry.stat().st_mtime < time.time() - 300:
                    os.remove(entry.path)
            except OSError:
                pass
            continue

        if entry.name.startswith("rrfs_"):
            try:
                mtime = entry.stat().st_mtime
                if mtime < cutoff:
                    os.remove(entry.path)
                else:
                    cached_entries.append((mtime, entry.path))
            except OSError:
                continue

    if len(cached_entries) > max_files:
        cached_entries.sort(key=lambda item: item[0])
        for _, old_path in cached_entries[: len(cached_entries) - max_files]:
            try:
                os.remove(old_path)
                with _RAM_CACHE_LOCK:
                    RAM_DATASET_CACHE.pop(old_path, None)
            except OSError:
                continue


def _check_idx_exists(url: str) -> str | None:
    try:
        req = urllib.request.Request(url + ".idx", method="HEAD")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                return url
    except Exception:
        return None


def resolve_cycle(run: str, fhour: str):
    """Fast parallel S3 cycle & filename discovery."""
    global PREFERRED_TEMPLATE

    ordered_templates = list(S3_TEMPLATES)
    if PREFERRED_TEMPLATE in ordered_templates:
        ordered_templates.remove(PREFERRED_TEMPLATE)
        ordered_templates.insert(0, PREFERRED_TEMPLATE)

    if run != "latest":
        ymd, hh = run.split("_")
        urls = [t.format(ymd=ymd, hh=hh, fhour=fhour) for t in ordered_templates]
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            for tmpl, url in zip(ordered_templates, pool.map(_check_idx_exists, urls)):
                if url:
                    PREFERRED_TEMPLATE = tmpl
                    return ymd, hh, url
        raise FileNotFoundError(f"RRFS {ymd} {hh}Z F{fhour} is not available on AWS S3 yet (or has aged out).")

    now = datetime.now(timezone.utc)
    cycle_hour = (now.hour // 6) * 6
    candidate = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

    for _ in range(6):
        ymd = candidate.strftime("%Y%m%d")
        hh = candidate.strftime("%H")
        urls = [t.format(ymd=ymd, hh=hh, fhour=fhour) for t in ordered_templates]
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            for tmpl, url in zip(ordered_templates, pool.map(_check_idx_exists, urls)):
                if url:
                    PREFERRED_TEMPLATE = tmpl
                    return ymd, hh, url
        candidate -= timedelta(hours=6)

    raise FileNotFoundError("Could not locate an available RRFS cycle on AWS S3.")


def get_static_orog(s3_url: str) -> xr.DataArray | None:
    """Download HGT:surface once from RRFS 2D/natlev f000, store as compact float32 NetCDF, and cache in RAM."""
    global CACHED_OROG_DA
    if CACHED_OROG_DA is not None:
        return CACHED_OROG_DA

    with _OROG_LOCK:
        if CACHED_OROG_DA is not None:
            return CACHED_OROG_DA

        if not (os.path.exists(OROG_PATH) and os.path.getsize(OROG_PATH) > 50_000):
            os.makedirs(DATA_DIR, exist_ok=True)
            base_dir, fname = s3_url.rsplit("/", 1)
            hh = fname.split(".")[1][1:3]
            domain = "na" if fname.endswith(".na.grib2") else "conus"

            candidate_urls = [
                f"{base_dir}/rrfs.t{hh}z.2dfld.3km.f000.{domain}.grib2",
                f"{base_dir}/rrfs.t{hh}z.2dfld.3km.f001.{domain}.grib2",
                f"{base_dir}/rrfs.t{hh}z.natlev.3km.f000.{domain}.grib2",
                f"{base_dir}/rrfs.t{hh}z.2dfld.3km.f000.conus.grib2",
                f"{base_dir}/rrfs.t{hh}z.2dfld.3km.f000.na.grib2",
                s3_url,
            ]

            downloaded = False
            for url in dict.fromkeys(candidate_urls):
                tmp_grib = os.path.join(tempfile.gettempdir(), f"orog_{os.getpid()}_{threading.get_ident()}.grib2")
                try:
                    idx_lines = urllib.request.urlopen(url + ".idx", timeout=10).read().decode("utf-8").splitlines()
                    records = []
                    for line in idx_lines:
                        parts = line.split(":")
                        if len(parts) >= 5:
                            records.append({"start": int(parts[1]), "var": parts[3].strip(), "level": parts[4].strip()})
                    records.sort(key=lambda r: r["start"])
                    for i in range(len(records) - 1):
                        records[i]["end"] = records[i + 1]["start"] - 1
                    records[-1]["end"] = None

                    orog_rec = next(
                        (r for r in records if r["var"] in {"HGT", "OROG"} and r["level"] == "surface"), None
                    )
                    if orog_rec is None:
                        continue

                    end_str = "" if orog_rec["end"] is None else str(orog_rec["end"])
                    req = urllib.request.Request(url, headers={"Range": f"bytes={orog_rec['start']}-{end_str}"})
                    with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_grib, "wb") as f_out:
                        f_out.write(resp.read())

                    with open(tmp_grib, "rb") as f:
                        gid = eccodes.codes_grib_new_from_file(f)
                        ny = eccodes.codes_get(gid, "Nj")
                        nx = eccodes.codes_get(gid, "Ni")
                        lats = eccodes.codes_get_array(gid, "latitudes").reshape(ny, nx).astype(np.float32)
                        lons = eccodes.codes_get_array(gid, "longitudes").reshape(ny, nx).astype(np.float32)
                        vals = eccodes.codes_get_values(gid).reshape(ny, nx).astype(np.float32)
                        eccodes.codes_release(gid)

                    lons = np.where(lons > 180, lons - 360, lons)
                    ds_orog = xr.Dataset(
                        {"orog": (("y", "x"), vals)},
                        coords={"latitude": (("y", "x"), lats), "longitude": (("y", "x"), lons)},
                    )
                    tmp_nc = f"{OROG_PATH}.{os.getpid()}_{threading.get_ident()}.tmp.nc"
                    ds_orog.to_netcdf(tmp_nc, encoding={"orog": {"zlib": True, "complevel": 1}})
                    os.replace(tmp_nc, OROG_PATH)
                    del ds_orog, lats, lons, vals
                    release_memory()
                    downloaded = True
                    print(f"[INFO] Cached static terrain ({os.path.getsize(OROG_PATH)/1e6:.1f} MB) from {url}")
                    break
                except Exception:
                    continue
                finally:
                    if os.path.exists(tmp_grib):
                        os.remove(tmp_grib)

            if not downloaded:
                print("[WARN] Could not fetch HGT:surface; falling back to 0m terrain.")
                return None

        ds_sfc = xr.open_dataset(OROG_PATH)
        CACHED_OROG_DA = ds_sfc["orog"].load()
        ds_sfc.close()
        return CACHED_OROG_DA


def fetch_s3_cropped(
    s3_url: str, ymd: str, hh: str, fhour: str, lat1: float, lon1: float, lat2: float, lon2: float
) -> str:
    """Download 16 pressure levels + 1000m AGL REFD via HTTP Range, decode/crop one GRIB message at a time via eccodes."""
    pad = 1.5
    top_lat = min(60.0, max(lat1, lat2) + pad)
    bot_lat = max(15.0, min(lat1, lat2) - pad)
    left_lon = max(-140.0, min(lon1, lon2) - pad)
    right_lon = min(-55.0, max(lon1, lon2) + pad)

    bbox_tag = f"{ymd}_{hh}z_f{fhour}_{bot_lat:.1f}_{top_lat:.1f}_{left_lon:.1f}_{right_lon:.1f}"
    nc_cache_path = os.path.join(CACHE_DIR, f"rrfs_s3crop_v3_{bbox_tag}.nc")

    if os.path.exists(nc_cache_path) and os.path.getsize(nc_cache_path) > 10_000:
        try:
            os.utime(nc_cache_path, None)
        except OSError:
            pass
        return nc_cache_path

    with get_fetch_lock(bbox_tag):
        if os.path.exists(nc_cache_path) and os.path.getsize(nc_cache_path) > 10_000:
            try:
                os.utime(nc_cache_path, None)
            except OSError:
                pass
            return nc_cache_path

        idx_lines = urllib.request.urlopen(s3_url + ".idx", timeout=10).read().decode("utf-8").splitlines()
        records = []
        for line in idx_lines:
            parts = line.split(":")
            if len(parts) >= 5:
                records.append({"start": int(parts[1]), "var": parts[3].strip(), "level": parts[4].strip()})
        records.sort(key=lambda r: r["start"])
        for i in range(len(records) - 1):
            records[i]["end"] = records[i + 1]["start"] - 1
        records[-1]["end"] = None

        wanted_vars = {"HGT", "UGRD", "VGRD", "TMP", "RWMR", "SNMR", "GRLE"}
        selected = []
        for r in records:
            if r["var"] in wanted_vars and r["level"].endswith(" mb"):
                try:
                    if int(float(r["level"].split()[0])) in TARGET_LEVELS_MB:
                        selected.append(r)
                except ValueError:
                    continue
            elif r["var"] == "REFD" and r["level"] == "1000 m above ground":
                selected.append(r)

        selected.sort(key=lambda r: r["start"])
        merged = []
        for r in selected:
            if merged and merged[-1]["end"] is not None and r["start"] == merged[-1]["end"] + 1:
                merged[-1]["end"] = r["end"]
            else:
                merged.append({"start": r["start"], "end": r["end"]})

        def download_chunk(chunk):
            end_str = "" if chunk["end"] is None else str(chunk["end"])
            req = urllib.request.Request(s3_url, headers={"Range": f"bytes={chunk['start']}-{end_str}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                return data[chunk["start"]: chunk["end"] + 1] if (resp.status == 200 and chunk["end"] is not None) else data

        tmp_grib = os.path.join(tempfile.gettempdir(), f"rrfs_tmp_{os.getpid()}_{threading.get_ident()}_{fhour}.grib2")
        try:
            with ThreadPoolExecutor(max_workers=6) as pool, open(tmp_grib, "wb") as f_out:
                for data_bytes in pool.map(download_chunk, merged):
                    f_out.write(data_bytes)
            release_memory()

            all_vars_3d = ("gh", "u", "v", "t", "rwmr", "snmr", "grle")
            levels_sorted = sorted(TARGET_LEVELS_MB, reverse=True)
            level_to_idx = {lvl: i for i, lvl in enumerate(levels_sorted)}

            y_slice = x_slice = None
            sub_lats = sub_lons = None
            arrays = {}
            refd_1km = None
            u_attrs = {}
            init_dt = valid_dt = None
            fhr_int = int(fhour)

            with open(tmp_grib, "rb") as f:
                while True:
                    gid = eccodes.codes_grib_new_from_file(f)
                    if gid is None:
                        break
                    try:
                        ny = eccodes.codes_get(gid, "Nj")
                        nx = eccodes.codes_get(gid, "Ni")

                        if y_slice is None:
                            lats_full = eccodes.codes_get_array(gid, "latitudes").reshape(ny, nx)
                            lons_full = eccodes.codes_get_array(gid, "longitudes").reshape(ny, nx)
                            lons_full = np.where(lons_full > 180, lons_full - 360, lons_full)

                            mask = (
                                (lats_full >= bot_lat) & (lats_full <= top_lat)
                                & (lons_full >= left_lon) & (lons_full <= right_lon)
                            )
                            y_idx, x_idx = np.where(mask)
                            y_slice = slice(int(y_idx.min()), int(y_idx.max()) + 1)
                            x_slice = slice(int(x_idx.min()), int(x_idx.max()) + 1)

                            sub_lats = lats_full[y_slice, x_slice].astype(np.float32)
                            sub_lons = lons_full[y_slice, x_slice].astype(np.float32)
                            del lats_full, lons_full, mask, y_idx, x_idx

                            sub_ny = y_slice.stop - y_slice.start
                            sub_nx = x_slice.stop - x_slice.start
                            for k in ("gh", "u", "v", "t"):
                                arrays[k] = np.full((len(levels_sorted), sub_ny, sub_nx), np.nan, dtype=np.float32)
                            for k in ("rwmr", "snmr", "grle"):
                                arrays[k] = np.zeros((len(levels_sorted), sub_ny, sub_nx), dtype=np.float32)
                            refd_1km = np.full((sub_ny, sub_nx), -20.0, dtype=np.float32)

                            data_date = str(eccodes.codes_get(gid, "dataDate"))
                            data_time = f"{int(eccodes.codes_get(gid, 'dataTime')):04d}"
                            init_dt = pd.to_datetime(f"{data_date}{data_time}", format="%Y%m%d%H%M")
                            fhr_int = int(eccodes.codes_get(gid, "forecastTime"))
                            valid_dt = init_dt + pd.Timedelta(hours=fhr_int)

                        type_of_level = eccodes.codes_get(gid, "typeOfLevel")
                        lvl = int(eccodes.codes_get(gid, "level"))

                        # Check for 1000 m AGL 2D reflectivity
                        if type_of_level == "heightAboveGround" and lvl == 1000:
                            vals_2d = eccodes.codes_get_values(gid).reshape(ny, nx)
                            sub_vals = vals_2d[y_slice, x_slice].astype(np.float32)
                            refd_1km = np.where((sub_vals < -30.0) | (sub_vals > 95.0) | np.isnan(sub_vals), -20.0, sub_vals)
                            del vals_2d, sub_vals
                            continue

                        if type_of_level != "isobaricInhPa" or lvl not in level_to_idx:
                            continue

                        short_name = eccodes.codes_get(gid, "shortName")
                        if short_name in {"gh", "u", "v", "t", "rwmr", "snmr", "grle"}:
                            vname = short_name
                        else:
                            param_cat = eccodes.codes_get(gid, "parameterCategory")
                            param_num = eccodes.codes_get(gid, "parameterNumber")
                            if (param_cat, param_num) == (3, 5):
                                vname = "gh"
                            elif (param_cat, param_num) == (0, 0):
                                vname = "t"
                            elif (param_cat, param_num) == (2, 2):
                                vname = "u"
                            elif (param_cat, param_num) == (2, 3):
                                vname = "v"
                            elif (param_cat, param_num) == (1, 22):
                                vname = "rwmr"
                            elif (param_cat, param_num) == (1, 25):
                                vname = "snmr"
                            elif (param_cat, param_num) == (1, 32):
                                vname = "grle"
                            else:
                                continue

                        if vname == "u" and not u_attrs:
                            u_attrs = {
                                "GRIB_LoVInDegrees": float(eccodes.codes_get(gid, "LoVInDegrees")),
                                "GRIB_LaDInDegrees": float(eccodes.codes_get(gid, "LaDInDegrees")),
                                "GRIB_Latin1InDegrees": float(eccodes.codes_get(gid, "Latin1InDegrees")),
                                "GRIB_Latin2InDegrees": float(eccodes.codes_get(gid, "Latin2InDegrees")),
                                "GRIB_uvRelativeToGrid": int(eccodes.codes_get(gid, "uvRelativeToGrid")),
                            }

                        vals_2d = eccodes.codes_get_values(gid).reshape(ny, nx)
                        sub_vals = vals_2d[y_slice, x_slice].astype(np.float32)
                        if vname in ("rwmr", "snmr", "grle"):
                            sub_vals = np.where((sub_vals < 0) | (sub_vals > 1.0) | np.isnan(sub_vals), 0.0, sub_vals)
                        arrays[vname][level_to_idx[lvl]] = sub_vals
                        del vals_2d, sub_vals
                    finally:
                        eccodes.codes_release(gid)

            data_vars = {k: (("isobaricInhPa", "y", "x"), arrays[k]) for k in all_vars_3d}
            data_vars["refd_1km"] = (("y", "x"), refd_1km)

            ds_crop = xr.Dataset(
                data_vars,
                coords={
                    "isobaricInhPa": np.array(levels_sorted, dtype=np.float32),
                    "latitude": (("y", "x"), sub_lats),
                    "longitude": (("y", "x"), sub_lons),
                },
                attrs={
                    "init_iso": init_dt.isoformat(),
                    "valid_iso": valid_dt.isoformat(),
                    "fhr": fhr_int,
                },
            )
            ds_crop["u"].attrs = u_attrs

            tmp_nc = f"{nc_cache_path}.{os.getpid()}_{threading.get_ident()}.tmp.nc"
            encoding = {v: {"zlib": True, "complevel": 1} for v in data_vars}
            ds_crop.to_netcdf(tmp_nc, encoding=encoding)
            ds_crop.close()
            del ds_crop, arrays, refd_1km, sub_lats, sub_lons
            os.replace(tmp_nc, nc_cache_path)
            release_memory()
            return nc_cache_path
        finally:
            if os.path.exists(tmp_grib):
                os.remove(tmp_grib)


def load_prepared_dataset(nc_path: str, orog_full: xr.DataArray | None) -> tuple:
    """Open cropped NetCDF, attach CF LambertConformal coordinates, and cache in RAM."""
    with _RAM_CACHE_LOCK:
        if nc_path in RAM_DATASET_CACHE:
            RAM_DATASET_CACHE.move_to_end(nc_path)
            return RAM_DATASET_CACHE[nc_path]

    ds_3d = xr.open_dataset(nc_path)

    attrs = ds_3d["u"].attrs
    rrfs_crs = ccrs.LambertConformal(
        central_longitude=attrs.get("GRIB_LoVInDegrees", -97.5),
        central_latitude=attrs.get("GRIB_LaDInDegrees", 38.5),
        standard_parallels=(
            attrs.get("GRIB_Latin1InDegrees", 38.5),
            attrs.get("GRIB_Latin2InDegrees", 38.5),
        ),
    )

    lons = xr.where(ds_3d["longitude"] > 180, ds_3d["longitude"] - 360, ds_3d["longitude"])
    lats = ds_3d["latitude"]
    xyz = rrfs_crs.transform_points(ccrs.PlateCarree(), lons.values, lats.values)

    if orog_full is None:
        orog_matched = np.zeros((ds_3d.sizes["y"], ds_3d.sizes["x"]), dtype=np.float32)
    elif ds_3d.sizes["y"] == orog_full.sizes["y"] and ds_3d.sizes["x"] == orog_full.sizes["x"]:
        orog_matched = orog_full.values
    elif "y" in ds_3d.coords and "x" in ds_3d.coords and "y" in orog_full.coords:
        orog_matched = orog_full.sel(y=ds_3d["y"], x=ds_3d["x"]).values
    else:
        full_lons = orog_full["longitude"].values
        full_lons = np.where(full_lons > 180, full_lons - 360, full_lons)
        full_lats = orog_full["latitude"].values
        dist0 = (full_lats - float(lats.values[0, 0])) ** 2 + (full_lons - float(lons.values[0, 0])) ** 2
        y0, x0 = np.unravel_index(np.argmin(dist0), dist0.shape)
        orog_matched = orog_full.values[y0 : y0 + ds_3d.sizes["y"], x0 : x0 + ds_3d.sizes["x"]]

    loaded_vars = [v for v in ["gh", "u", "v", "t", "rwmr", "snmr", "grle", "refd_1km"] if v in ds_3d]
    ds = ds_3d[loaded_vars].load()
    ds["orog"] = (("y", "x"), orog_matched)
    ds = ds.assign_coords(
        x=("x", xyz[0, :, 0], {"standard_name": "projection_x_coordinate", "units": "m"}),
        y=("y", xyz[:, 0, 1], {"standard_name": "projection_y_coordinate", "units": "m"}),
        longitude=lons,
        latitude=lats,
    ).metpy.assign_crs(rrfs_crs.to_cf()).metpy.parse_cf()

    meta = {
        "crs": rrfs_crs,
        "uv_grid_rel": attrs.get("GRIB_uvRelativeToGrid", 0),
        "init_dt": pd.to_datetime(ds_3d.attrs["init_iso"]),
        "valid_dt": pd.to_datetime(ds_3d.attrs["valid_iso"]),
        "fhr": int(ds_3d.attrs["fhr"]),
    }
    ds_3d.close()

    with _RAM_CACHE_LOCK:
        RAM_DATASET_CACHE[nc_path] = (ds, meta)
        while len(RAM_DATASET_CACHE) > MAX_RAM_ITEMS:
            RAM_DATASET_CACHE.popitem(last=False)
    release_memory()

    return ds, meta


def compute_thompson_dbz(p_mb: np.ndarray, temp_k: np.ndarray, rwmr: np.ndarray, snmr: np.ndarray, grle: np.ndarray) -> np.ndarray:
    """Compute equivalent radar reflectivity (dBZ) from Thompson microphysics species (rain, snow, graupel)."""
    p_pa = p_mb * 100.0
    rho = p_pa / (287.05 * np.maximum(temp_k, 150.0))

    qr = np.maximum(rwmr, 0.0)
    qs = np.maximum(snmr, 0.0)
    qg = np.maximum(grle, 0.0)

    z_rain = 3.63e9 * np.power(rho * qr, 1.75)
    snow_coeff = np.where(temp_k > 273.15, 4.26e11, 9.80e8)
    z_snow = snow_coeff * np.power(rho * qs, 1.75)
    z_graupel = 4.33e10 * np.power(rho * qg, 1.75)

    z_total = z_rain + z_snow + z_graupel
    return 10.0 * np.log10(np.maximum(z_total, 1e-3))


def prefetch_next_hour(run: str, next_fhour: int, lat1: float, lon1: float, lat2: float, lon2: float):
    """Silently warm the disk cache for fhour+1 in the background (without bloating RAM)."""
    if next_fhour > 84:
        return
    try:
        fhour_str = f"{next_fhour:03d}"
        ymd, hh, s3_url = resolve_cycle(run=run, fhour=fhour_str)
        fetch_s3_cropped(s3_url, ymd, hh, fhour_str, lat1, lon1, lat2, lon2)
    except Exception:
        pass


@app.on_event("startup")
def warm_cache_on_startup():
    def _warmup():
        try:
            ymd, hh, s3_url = resolve_cycle(run="latest", fhour="001")
            get_static_orog(s3_url)
            print(f"[STARTUP] Warmed RRFS {ymd} {hh}Z metadata & static terrain.")
        except Exception as e:
            print(f"[STARTUP] Warmup skipped: {e}")

    threading.Thread(target=_warmup, daemon=True).start()


@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "ok", "service": "RRFS 3km Cross-Section API"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


@app.get("/api/cross_section")
def generate_cross_section(
    background_tasks: BackgroundTasks,
    lat1: float = Query(...),
    lon1: float = Query(...),
    lat2: float = Query(...),
    lon2: float = Query(...),
    ymax: int = Query(10000),
    fhour: int = Query(1),
    run: str = Query("latest"),
):
    with RENDER_SEMAPHORE:
        try:
            t0 = time.perf_counter()
            os.makedirs(CACHE_DIR, exist_ok=True)
            prune_cache(CACHE_DIR, max_age_hours=CACHE_MAX_AGE_HOURS, max_files=MAX_CACHE_FILES)
            fhour_str = f"{fhour:03d}"

            ymd, hh, s3_url = resolve_cycle(run=run, fhour=fhour_str)
            orog_full = get_static_orog(s3_url)
            t1 = time.perf_counter()

            nc_path = fetch_s3_cropped(s3_url, ymd, hh, fhour_str, lat1, lon1, lat2, lon2)
            ds, meta = load_prepared_dataset(nc_path, orog_full)
            t2 = time.perf_counter()

            # Subset only 3D fields for MetPy cross_section interpolation
            ds_3d_sub = ds[["gh", "u", "v", "t", "rwmr", "snmr", "grle", "orog"]]
            cross = cross_section(ds_3d_sub, (lat1, lon1), (lat2, lon2), steps=150).set_coords(("latitude", "longitude"))

            if meta["uv_grid_rel"] == 1:
                c_xyz = meta["crs"].transform_points(
                    ccrs.PlateCarree(), cross["longitude"].values, cross["latitude"].values
                )
                X_proj = np.broadcast_to(c_xyz[:, 0], cross["u"].shape)
                Y_proj = np.broadcast_to(c_xyz[:, 1], cross["u"].shape)
                u_earth, v_earth = ccrs.PlateCarree().transform_vectors(
                    meta["crs"], X_proj, Y_proj, cross["u"].values, cross["v"].values
                )
            else:
                u_earth, v_earth = cross["u"].values, cross["v"].values

            wind_spd = mpcalc.wind_speed(u_earth * units("m/s"), v_earth * units("m/s")).to("kt").magnitude
            temp_k = cross["t"].values
            temp_c = (temp_k - 273.15) * (9 / 5) + 32  # in Fahrenheit

            geod = Geod(ellps="WGS84")
            _, _, dist_m = geod.inv(
                np.full(cross.sizes["index"], cross["longitude"].values[0]),
                np.full(cross.sizes["index"], cross["latitude"].values[0]),
                cross["longitude"].values,
                cross["latitude"].values,
            )
            dist_km = dist_m / 1000.0
            X_2d = np.broadcast_to(dist_km, cross["gh"].shape)
            Z_2d = cross["gh"].values
            P_2d = np.broadcast_to(cross["isobaricInhPa"].values[:, None], Z_2d.shape)

            rwmr_val = cross["rwmr"].values if "rwmr" in cross else np.zeros_like(temp_k)
            snmr_val = cross["snmr"].values if "snmr" in cross else np.zeros_like(temp_k)
            grle_val = cross["grle"].values if "grle" in cross else np.zeros_like(temp_k)
            dbz_2d = compute_thompson_dbz(P_2d, temp_k, rwmr_val, snmr_val, grle_val)

            # Single-column 4-panel vertical stack (3 cross-sections + 1 2D 1km reflectivity map)
            fig = Figure(figsize=(12, 18.5), dpi=115)
            gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.25], hspace=0.28)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
            ax_map = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())

            above_ground = Z_2d >= cross["orog"].values[None, :]
            visible_mask = (Z_2d <= ymax) & above_ground

            # --- Panel 1: Wind Speed (kt), Barbs & Isobars ---
            max_spd = np.nanmax(wind_spd[visible_mask])
            vmax = max(10.0, np.ceil(max_spd / 5.0) * 5.0)
            cf1 = ax1.contourf(X_2d, Z_2d, wind_spd, levels=np.linspace(0, vmax, 17), cmap="plasma", extend="neither")
            fig.colorbar(cf1, ax=ax1, pad=0.02, label="Wind Speed (kt)")

            p_levels = [250, 300, 400, 500, 600, 650, 700, 750, 800, 850, 925, 1000]
            cs_p1 = ax1.contour(
                X_2d, Z_2d, P_2d, levels=p_levels,
                colors="white", linestyles="--", linewidths=0.9, alpha=0.85,
            )
            ax1.clabel(cs_p1, fmt="%d hPa", fontsize=8, inline=True)

            skip_x = max(1, len(dist_km) // 20)
            skip_z = 2 if ymax > 5000 else 1
            u_kt_masked = np.where(above_ground, u_earth * 1.94384, np.nan)
            v_kt_masked = np.where(above_ground, v_earth * 1.94384, np.nan)
            ax1.barbs(
                X_2d[::skip_z, ::skip_x], Z_2d[::skip_z, ::skip_x],
                u_kt_masked[::skip_z, ::skip_x], v_kt_masked[::skip_z, ::skip_x],
                length=6, color="black", alpha=0.85, zorder=5,
            )
            ax1.fill_between(dist_km, 0, cross["orog"].values, color="#5c4033", edgecolor="black", zorder=10, label="Terrain")
            ax1.set_ylabel("Geopotential Height (m ASL)")
            ax1.set_title("Wind Speed (kt), Wind Barbs & Isobars")
            ax1.legend(loc="upper right")

            # --- Panel 2: Air Temperature (°F) & 32°F Freezing Level ---
            tmin = np.floor(np.nanmin(temp_c[visible_mask]) / 10.0) * 10.0
            tmax = np.ceil(np.nanmax(temp_c[visible_mask]) / 10.0) * 10.0
            cf2 = ax2.contourf(X_2d, Z_2d, temp_c, levels=np.arange(tmin, tmax + 5.0, 5.0), cmap="RdYlBu_r", extend="both")
            fig.colorbar(cf2, ax=ax2, pad=0.02, label="Temperature (°F)")

            cs_0 = ax2.contour(X_2d, Z_2d, temp_c, levels=[32], colors="black", linewidths=2.0, zorder=6)
            ax2.clabel(cs_0, fmt="32°F", fontsize=9, inline=True)
            cs_p2 = ax2.contour(
                X_2d, Z_2d, P_2d, levels=p_levels,
                colors="dimgray", linestyles="--", linewidths=0.9, alpha=0.85,
            )
            ax2.clabel(cs_p2, fmt="%d hPa", fontsize=8, inline=True)
            ax2.fill_between(dist_km, 0, cross["orog"].values, color="#5c4033", edgecolor="black", zorder=10, label="Terrain")
            ax2.set_ylabel("Geopotential Height (m ASL)")
            ax2.set_title("Air Temperature (°F) & 32°F Freezing Level")
            ax2.legend(loc="upper right")

            # --- Panel 3: 3D Simulated Reflectivity (dBZ, Thompson Microphysics) ---
            dbz_masked = np.ma.masked_where((~above_ground) | (dbz_2d < 5.0), dbz_2d)
            cf3 = ax3.contourf(
                X_2d, Z_2d, dbz_masked,
                levels=DBZ_LEVELS, cmap=DBZ_CMAP, norm=DBZ_NORM, extend="max",
            )
            fig.colorbar(cf3, ax=ax3, pad=0.02, ticks=DBZ_LEVELS, label="Simulated Reflectivity (dBZ)")

            cs_0_ref = ax3.contour(X_2d, Z_2d, temp_c, levels=[32], colors="black", linestyles=":", linewidths=1.5, zorder=6)
            ax3.clabel(cs_0_ref, fmt="32°F", fontsize=8, inline=True)
            cs_p3 = ax3.contour(
                X_2d, Z_2d, P_2d, levels=p_levels,
                colors="dimgray", linestyles="--", linewidths=0.9, alpha=0.75,
            )
            ax3.clabel(cs_p3, fmt="%d hPa", fontsize=8, inline=True)
            ax3.fill_between(dist_km, 0, cross["orog"].values, color="#5c4033", edgecolor="black", zorder=10, label="Terrain")

            ax3.set_ylim(0, ymax)
            ax3.set_xlim(0, dist_km[-1])
            ax3.set_xlabel(f"Distance Along Transect (km)  |  Start A: ({lat1:.3f}, {lon1:.3f}) → End B: ({lat2:.3f}, {lon2:.3f})")
            ax3.set_ylabel("Geopotential Height (m ASL)")
            ax3.set_title("3D Simulated Radar Reflectivity (dBZ, Thompson Microphysics)")
            ax3.legend(loc="upper right")

            # --- Panel 4: 2D 1 km AGL Simulated Reflectivity Map + Cross-Section Line ---
            map_lons = ds["longitude"].values
            map_lats = ds["latitude"].values
            refd_map = ds["refd_1km"].values if "refd_1km" in ds else np.full_like(map_lats, -20.0)
            refd_map_masked = np.ma.masked_where(refd_map < 5.0, refd_map)

            ax_map.set_facecolor("#f4f6f8")
            ax_map.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eef0f2", zorder=0)
            ax_map.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dce8f2", zorder=0)
            ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="#444444", zorder=4)
            ax_map.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.7, edgecolor="#666666", zorder=4)

            cf_map = ax_map.pcolormesh(
                map_lons, map_lats, refd_map_masked,
                cmap=DBZ_CMAP, norm=DBZ_NORM, shading="auto", transform=ccrs.PlateCarree(), zorder=3,
            )
            fig.colorbar(
                cf_map, ax=ax_map, pad=0.02, ticks=DBZ_LEVELS, label="1 km AGL Reflectivity (dBZ)",
            )

            ax_map.plot(
                [lon1, lon2], [lat1, lat2],
                color="black", linewidth=3.4, transform=ccrs.Geodetic(), zorder=6,
            )
            ax_map.plot(
                [lon1, lon2], [lat1, lat2],
                color="#ff1744", linewidth=2.0, linestyle="--", transform=ccrs.Geodetic(), zorder=7,
                label="Cross-Section Transect",
            )
            ax_map.scatter(
                [lon1, lon2], [lat1, lat2],
                c=["#00e676", "#ff1744"], s=70, edgecolors="black", linewidths=1.3,
                transform=ccrs.PlateCarree(), zorder=8,
            )
            ax_map.text(
                lon1, lat1, "  A (Start)", fontsize=9.5, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
                transform=ccrs.PlateCarree(), zorder=9,
            )
            ax_map.text(
                lon2, lat2, "  B (End)", fontsize=9.5, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
                transform=ccrs.PlateCarree(), zorder=9,
            )

            pad_view = 1.35
            ax_map.set_extent(
                [
                    max(-139.5, min(lon1, lon2) - pad_view),
                    min(-55.5, max(lon1, lon2) + pad_view),
                    max(15.5, min(lat1, lat2) - pad_view),
                    min(59.5, max(lat1, lat2) + pad_view),
                ],
                crs=ccrs.PlateCarree(),
            )
            gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.45, linestyle=":")
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8.5}
            gl.ylabel_style = {"size": 8.5}
            ax_map.set_title("1 km AGL Reflectivity & Transect Location")
            ax_map.legend(loc="lower right", framealpha=0.9)

            fig.suptitle(
                f"RRFS 3km Cross-Section  |  Init: {meta['init_dt']:%Y-%m-%d %HZ} (F{meta['fhr']:03d})  |  Valid: {meta['valid_dt']:%Y-%m-%d %H:%M UTC}",
                fontsize=13, fontweight="bold",
            )
            fig.subplots_adjust(left=0.08, right=0.95, top=0.945, bottom=0.04)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=115)
            fig.clear()
            buf.seek(0)
            png_bytes = buf.read()
            buf.close()
            del fig, ax1, ax2, ax3, ax_map, cross, wind_spd, temp_c, temp_k, dbz_2d, dbz_masked, refd_map, refd_map_masked, u_earth, v_earth
            release_memory()

            t3 = time.perf_counter()
            print(f"[TIMING] Resolve: {t1-t0:.2f}s | Fetch+Crop: {t2-t1:.2f}s | Slice+Plot: {t3-t2:.2f}s")

            background_tasks.add_task(prefetch_next_hour, run, fhour + 1, lat1, lon1, lat2, lon2)

            return Response(content=png_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})

        except Exception:
            release_memory()
            raise HTTPException(status_code=500, detail=traceback.format_exc())