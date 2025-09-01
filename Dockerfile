FROM python:3.11-slim

WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8082

# --host 0.0.0.0 é essencial no Docker; --reload só em dev
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]