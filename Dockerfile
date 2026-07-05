FROM python:3.11-slim

WORKDIR /app

COPY requirements-gateway.txt .
RUN pip install --no-cache-dir -r requirements-gateway.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
