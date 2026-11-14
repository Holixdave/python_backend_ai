FROM python:3.13-slim

# ffmpeg for video/audio work inside the sandbox. The image's own Python (3.13) is the ONLY
# interpreter the sandbox uses (gpt2_sandbox.py runs user code with sys.executable), so
# numpy/pillow/imageio below are always importable without any runtime pip install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Free self-hosted TTS: Piper binary + 4 voices (replaces ElevenLabs).
RUN curl -fsSL "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz" | tar -xz -C /opt
RUN mkdir -p /app/models && for v in \
      en/en_US/ryan/medium/en_US-ryan-medium \
      en/en_US/joe/medium/en_US-joe-medium \
      en/en_GB/alan/medium/en_GB-alan-medium \
      en/en_US/amy/medium/en_US-amy-medium; do \
    n=$(basename "$v"); \
    curl -fsSL "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx" -o "/app/models/$n.onnx" && \
    curl -fsSL "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx.json" -o "/app/models/$n.onnx.json"; \
  done

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the small background-removal model once at build time, so no request ever waits for it.
ENV U2NET_HOME=/app/.u2net
RUN python -c "from rembg import new_session; new_session('u2netp')" || echo "model preload skipped"

COPY . .

# Bundled fonts (fonts/FONTS.txt lists them). gpt2_sandbox.py passes this path to run_code so PDF code
# can load e.g. Poppins/Lora; see fonts/pdf_fonts.py.
ENV FONTS_DIR=/app/fonts

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
