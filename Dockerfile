# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt
COPY preprocess/ /app/preprocess/
COPY models/ /app/models/
COPY results/ /app/results/

# Install system dependencies (for audio processing) and streamlit app support
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
