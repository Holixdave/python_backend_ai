FROM python:3.13-slim

# ffmpeg for video/audio work inside the sandbox. The image's own Python (3.13) is the ONLY
# interpreter the sandbox uses (gpt2_sandbox.py runs user code with sys.executable), so
# numpy/pillow/imageio below are always importable without any runtime pip install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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
