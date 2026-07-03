# Podman/Docker-compatible build file (standard Dockerfile syntax).
#   podman build -t polardrl .   (Fedora default — no Docker Engine needed)
#   docker build -t polardrl .   (works identically on Docker if a teammate uses it)
FROM python:3.11-slim

WORKDIR /app

# uv gives fast, reproducible installs from pyproject.toml
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/

RUN uv pip install --system .

CMD ["python", "-c", "import polardrl; print('PolarDRL reproducibility image OK — Python', __import__('sys').version)"]
