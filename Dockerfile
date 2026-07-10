# Use a standard Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install compilation tools needed for compiling llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first to cache package installation step
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY src/ ./src/

# Copy the local model into the image as required by the guide
# Note: Ensure you have model.gguf downloaded in the repository root before building.
COPY model.gguf ./model.gguf

# Set the entrypoint to execute the smart router script
CMD ["python", "src/test_main.py"]
