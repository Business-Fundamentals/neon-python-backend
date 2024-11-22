# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install system and git dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    uild-essential \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    git \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Copy the application files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Git LFS
RUN git lfs install

# Clone the LaBSE repository from Hugging Face
RUN git clone https://huggingface.co/sentence-transformers/LaBSE /app/models/LaBSE

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
