FROM condaforge/miniforge3:latest

WORKDIR /app

# Install geospatial & meteorological stack via conda-forge (includes eccodes, proj, geos)
RUN conda install -y -c conda-forge \
    python=3.11 \
    fastapi \
    uvicorn \
    xarray \
    cfgrib \
    eccodes \
    metpy \
    cartopy \
    matplotlib \
    scipy \
    netcdf4 \
    requests && \
    conda clean -afy

COPY app.py /app/app.py
RUN mkdir -p /app/Data/cache

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]