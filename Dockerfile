# Use a lightweight official Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure stdout/stderr are unbuffered
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Set working directory
WORKDIR /app

# Install system dependencies (optional but useful for many ML packages)
# RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        build-essential \
#        gcc && \
#    rm -rf /var/lib/apt/lists/*

# chngs
RUN apt-get update && \
    apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the application
CMD ["python", "-m", "app.app"]


# docker build --no-cache -t predictive-maintenance .
# docker run -p 5000:5000 predictive-maintenance