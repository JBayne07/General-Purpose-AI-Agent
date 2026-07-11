# Use a standard Python image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Copy the rest of the project files
COPY src/ ./src/

RUN mkdir -p /input /output

# Copy the local model into the image
COPY model.gguf ./model.gguf

# Set the entrypoint to execute the smart router script
CMD ["python3", "-m", "src.main"]
