FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    SDR_MODEL_REPO=slicedonions/self-driving-rl-v1 \
    SDR_MAX_SESSIONS=6

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && python -m pip install \
        "self-driving-rl @ git+https://github.com/becauseno7/Self-Driving-RL.git@v1.0.0" \
        fastapi==0.141.1 \
        "uvicorn[standard]==0.52.1" \
        huggingface_hub==1.26.0

USER user
WORKDIR /home/user/app

COPY --chown=user:user app.py ./app.py
COPY --chown=user:user static ./static

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

