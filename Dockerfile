# Base image with Ubuntu and CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.11 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    wget \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set up Poetry
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ensure Poetry is in the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the PYTHONPATH to include the /app directory
ENV PYTHONPATH=/app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock* /app/
WORKDIR /app

# Install Python dependencies via Poetry
RUN poetry install --no-root

# Copy the application code
COPY . /app

# Expose the port for the FastAPI app
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["poetry", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
