# Start from a minimal Python image. "slim" drops build tools we do not need, which keeps the final image small.
FROM python:3.12-slim

WORKDIR /app

# Copy requirements first, before the code. Docker caches each step, so as long
# as this file does not change, the slow pip install step is reused even when
# the source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs: the API code and the exported model.
COPY src/__init__.py src/__init__.py
COPY src/api/ src/api/
COPY artifacts/model.onnx artifacts/model.onnx

# Documents which port the app listens on.
EXPOSE 8000

# The command that runs when the container starts.
# host 0.0.0.0 makes the server reachable from outside the container.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]