# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app


# Install system dependencies
RUN apt update && \
    apt install -y libvips-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .


# Expose the port the app runs on (change if using a different port)
EXPOSE 8000

# Command to run the application (assuming using FastAPI with uvicorn)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
