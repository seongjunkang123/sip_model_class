# Use an official Python runtime as the base image
FROM python:3.11.9-slim

# Create and set the working directory inside the container
WORKDIR /

# Copy only the requirements file first (for efficient caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Command to run your app (change as needed)
CMD ["python", "main.py"]