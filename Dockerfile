# AirfRANS GNN API. Built by Hugging Face Spaces (Docker SDK) on every deploy,
# and by CI for the smoke test. Run locally:
#   docker build -t airfrans-gnn .
#   docker run -p 7860:7860 airfrans-gnn                  # downloads artifacts from the Hub
#   docker run -p 7860:7860 -v $PWD/serving_artifacts:/home/user/artifacts airfrans-gnn
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Spaces run the container as uid 1000; creating that user keeps file
# ownership the same locally and on the Space
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# dependencies first, so code-only changes reuse these cached layers
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY serve/requirements.txt serve/requirements.txt
RUN pip install -r serve/requirements.txt

COPY pyproject.toml ./
COPY src src
RUN pip install --no-deps .
COPY serve serve

USER user
ENV ARTIFACT_DIR=/home/user/artifacts \
    HF_MODEL_REPO=tingluzhang/airfrans-gnn \
    HF_HOME=/home/user/.cache/huggingface
EXPOSE 7860
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
