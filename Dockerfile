#official Python runtime as a parent image
FROM python:3.9-slim

#working directory in the container
WORKDIR /app

# Copy the requirements file to Docker cache
COPY requirements.txt .

# Install any needed packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 80
EXPOSE 80

# Run the app using gunicorn
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:80", "app:app"]