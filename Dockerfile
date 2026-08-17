# Stage 1: build the React dashboard
FROM node:22 AS dashboard-build
WORKDIR /app/dashboard-ui
COPY dashboard-ui/package.json dashboard-ui/package-lock.json ./
RUN npm ci
COPY dashboard-ui/ ./
RUN npm run build

# Stage 2: Python runtime (shared by api + worker services)
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12
COPY . .
COPY --from=dashboard-build /app/dashboard/dist ./dashboard/dist