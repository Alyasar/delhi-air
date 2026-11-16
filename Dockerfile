# Small official Python image, same version as development
FROM python:3.12-slim

# No .pyc files; print logs immediately (so the host shows them in real time)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilt when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The application and the trained model (nothing else: see .dockerignore)
COPY app/ app/
COPY models/ models/

# Run as a normal user, not root
RUN useradd --create-home appuser
USER appuser

# Documented port (cloud hosts override it with $PORT)
EXPOSE 8000

# Docker checks every 30 s that the app answers on /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health')" || exit 1

# Start the server on all interfaces, on $PORT if the host sets it, else 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]