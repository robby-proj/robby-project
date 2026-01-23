FROM python:3.11-slim

# System deps for OpenCV video capture
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY models /app/models

ENV MODEL_PATH=/app/models/tmp/detect.tflite
ENV LABELS_PATH=/app/models/tmp/labelmap.txt

CMD ["python", "-u", "app.py"]
