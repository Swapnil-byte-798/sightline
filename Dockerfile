# Sightline API image.
#
# Multi-stage, non-root, and designed to run with a read-only root filesystem.
# The reasoning is the same one that runs through the rest of the project: the
# process holds an index that is a denormalised copy of who may read what, so
# the blast radius of a compromise is the corpus. Give it as little as it can
# work with.
#
# What this image deliberately does NOT contain:
#
#   * a compiler or build toolchain — they exist in the builder stage only,
#   * the optional extras (Qdrant client, psycopg, onnxruntime, faiss) — the
#     default backends are in-process, and pulling ~1 GB of wheels to run a
#     demo is a cost nobody agreed to. Build with
#     `--build-arg EXTRAS=".[qdrant,postgres,embed]"` when you want them,
#   * the test suite, the evaluation harness, and the docs.
#
# Pinned to 3.12 rather than 3.13: that is what CI's matrix covers.

# --------------------------------------------------------------------------
# Stage 1 — build a virtualenv
# --------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ARG EXTRAS="."

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed by some optional extras and by nothing in the
# default install. It stays in this stage and is never copied forward.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency metadata first, so a source-only change does not re-resolve wheels.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
 && pip install "${EXTRAS}"

# --------------------------------------------------------------------------
# Stage 2 — runtime
# --------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="sightline" \
      org.opencontainers.image.description="Internal AI search that can only see what you can see." \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Where a read-only rootfs still lets us write. Both are tmpfs mounts in
    # docker-compose.yml; if you run this image by hand, mount them yourself.
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    SIGHTLINE_ENV=production

# curl for the healthcheck, and nothing else. An image with no HTTP client
# cannot health-check itself, and an image with a shell full of tools is a
# foothold.
RUN apt-get update \
 && apt-get install --no-install-recommends -y curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin sightline

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER 10001:10001

EXPOSE 8000

# /healthz is liveness: the process is up. /readyz is readiness and checks that
# the tuple store's epoch is readable, because a server that cannot read the
# policy must not be sent traffic — it would have to choose between refusing and
# guessing, and only one of those is safe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# No --reload, no --workers: one process per container, scaled by the
# orchestrator. Workers here would give each its own plan cache and make
# "which epoch served this request" a per-worker question.
CMD ["uvicorn", "sightline.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-server-header", "--proxy-headers"]
