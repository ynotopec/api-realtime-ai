FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends ffmpeg \ 
    && python -m venv ${VENV_PATH} \ 
    && ${VENV_PATH}/bin/pip install --upgrade pip \ 
    && rm -rf /var/lib/apt/lists/*

ENV PATH="${VENV_PATH}/bin:${PATH}"
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--ws", "websockets"]
