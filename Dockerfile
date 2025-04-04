FROM python:3.11

# Set working directory
WORKDIR /app

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port
EXPOSE 5050

# Run the app
CMD ["python", "app.py"]
