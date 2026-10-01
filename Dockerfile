FROM python:3.9-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY config/ ./config/
COPY scripts/verify_model_artifacts.py ./scripts/verify_model_artifacts.py

RUN python scripts/verify_model_artifacts.py

WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]