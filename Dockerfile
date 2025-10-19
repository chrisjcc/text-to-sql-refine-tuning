# Multi-stage build for optimized image size
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs cache data_cache outputs

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python3", "-m", "src.inference.api", "--model-path", "./outputs/final_model", "--host", "0.0.0.0", "--port", "8000"]
