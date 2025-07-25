FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set the default command to run cam_detector.py
CMD ["python", "cam_detector.py"]
