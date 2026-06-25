FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the model (runs once during build)
COPY download_model.py .
RUN python download_model.py

# Copy source files
COPY rank.py .
COPY config.yaml .
COPY jd.txt .
COPY validate_submission.py .

CMD ["python", "rank.py", \
     "--candidates", "/data/candidates.jsonl", \
     "--out", "/data/submission.csv"]