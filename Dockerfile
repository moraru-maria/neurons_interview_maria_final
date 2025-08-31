# Use Python 3.11+ 
FROM python:3.11-slim

# Keep Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Create app folder
WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose API port
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
