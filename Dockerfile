FROM python:3.12-slim

WORKDIR /app

# Copy source
COPY rank.py .
COPY requirements.txt .
COPY validate_submission.py .

# No pip install needed — stdlib only
# This just documents the runtime environment

CMD ["python", "rank.py", \
     "--candidates", "/data/candidates.jsonl", \
     "--out", "/data/submission.csv"]
