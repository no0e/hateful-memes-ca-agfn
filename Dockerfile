# A CPU image that trains and evaluates CA-AGFN with nothing installed locally.
#
#   docker build -t ca-agfn .
#   docker run --rm ca-agfn                                   # smoke run, fake data
#   docker run --rm -v $PWD/data:/app/data -v $PWD/results:/app/results \
#       ca-agfn python scripts/train.py --variant full --seed 0
#
# GPU training uses the same code on a CUDA base image or straight on the host;
# this image is for reproducing the pipeline anywhere, and for CI.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/hf
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY ca_agfn ca_agfn
RUN pip install --no-cache-dir ".[figures,track]"
COPY scripts scripts
COPY weights weights

RUN useradd --create-home runner && chown -R runner /app
USER runner
CMD ["python", "scripts/train.py", "--smoke", "--results", "/tmp/smoke"]
