FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/api/ ./src/api/
COPY src/__init__.py ./src/__init__.py
COPY models/ ./models/
COPY data/raw/ ./data/raw/
COPY data/processed/ ./data/processed/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
