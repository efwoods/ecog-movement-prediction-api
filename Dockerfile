# Use official Python 3.11 slim image for lightweight base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first for caching dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code into the container
COPY src ./src

# Expose port 8000 (adjust if your app uses a different port)
EXPOSE 8000

# Set default command to run your API (adjust entry point as needed)
CMD ["python", "src/main.py"]
