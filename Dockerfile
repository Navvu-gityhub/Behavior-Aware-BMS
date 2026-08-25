# Multi-stage build. The wheel-building stage carries a compiler for any
# source distribution in the dependency set; the runtime stage does not, which
# keeps a C toolchain out of the shipped image.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.12-slim AS runtime

# joblib determines the physical core count by shelling out to a platform
# command, which fails inside minimal containers and can abort a run. Every
# estimator in this repo is configured n_jobs=1, so pinning this only skips a
# probe whose answer is unused. See src/bms/benchmarks/classical.py.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOKY_MAX_CPU_COUNT=1

# Run as a non-root user. The container writes generated CSVs under
# data/features/ and a dashboard at the repo root, so the working tree must be
# owned by that user rather than made world-writable.
RUN useradd --create-home --uid 10001 bms

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=bms:bms . .

USER bms

EXPOSE 8000

# The API's /healthz reports service liveness and is deliberately NOT battery
# health — see src/bms/api/app.py.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.bms.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
