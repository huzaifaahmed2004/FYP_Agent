# Lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build essentials for some wheels and clean up apt cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    # Install CPU-only PyTorch to avoid pulling massive CUDA images
    && pip install --no-cache-dir --prefer-binary \
       torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
       -f https://download.pytorch.org/whl/cpu \
    # Install remaining requirements
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
