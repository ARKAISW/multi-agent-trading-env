# Stage 1: Build the React Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
RUN npm run build

# Stage 2: Final Image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_LOCAL_POLICY=false
ENV LOCAL_MODEL_PATH=/app/models/local_policy
ENV ENABLE_REMOTE_PM=false
ENV ENABLE_REMOTE_JUDGE=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-space.txt ./
RUN pip install --no-cache-dir -r requirements-space.txt

# Copy everything else (including backend and any local models)
COPY . .

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/ui/dist ./ui/dist

# Ensure model directory exists (even if empty, for safety)
RUN mkdir -p models/local_policy

# Hugging Face Spaces use port 7860 by default
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
