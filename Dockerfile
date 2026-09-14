# Serves the Gradio inference app. The model itself is not baked into the
# image — it is downloaded from Hugging Face Hub at container startup, so
# this image stays small and works on any host without Google Drive access.

FROM python:3.11-slim

WORKDIR /app

# System dependency required by some tokenizer/audio backends pulled in by
# transformers; harmless to keep even if unused.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV MODEL_DIR="rahaf088/arabic-hate-speech-marbert-lora"
EXPOSE 7860

CMD ["sh", "-c", "python src/app.py --model-dir ${MODEL_DIR}"]
