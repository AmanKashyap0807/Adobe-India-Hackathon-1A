FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY model.txt .

# Set the default command
CMD ["python", "src/infer/predict.py", "/app/input/input.pdf"] 