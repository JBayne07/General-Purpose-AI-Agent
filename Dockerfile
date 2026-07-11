# Use a standard Python image
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 2. Install build-essential, python3-dev, AND cmake (required for llama-cpp-python compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 3. Disable CPU instructions that are not universally supported in cloud VMs
ENV CMAKE_ARGS="-DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_FMA=OFF -DLLAMA_F16C=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF" \
    FORCE_CMAKE=1

# Set the working directory
WORKDIR /app

# Copy requirements file first to cache package installation step
COPY requirements.txt .

# Install dependencies (will compile llama-cpp-python with the CMAKE_ARGS above)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY src/ ./src/

# Copy the local model into the image
COPY model.gguf ./model.gguf

# Set the entrypoint to execute the smart router script
CMD ["python3", "src/main.py"]
