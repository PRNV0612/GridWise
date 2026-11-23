FROM python:3.11-slim

WORKDIR /app

# libgomp1: LightGBM's OpenMP runtime (not in slim images)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY gridwise ./gridwise

RUN pip install --no-cache-dir -e . uvicorn[standard]

# Models and the processed parquet are gitignored artifacts: mount them at
# runtime (docker-compose.yml does) rather than baking 2 GB into the image.
# Build them first with:
#   .venv/bin/python -m gridwise.etl && .venv/bin/python -m gridwise.train \
#   && .venv/bin/python -m gridwise.anomalies
EXPOSE 8000

CMD ["uvicorn", "gridwise.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
