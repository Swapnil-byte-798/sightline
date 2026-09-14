# Sightline API image.
#
# Multi-stage: the build stage carries compilers and caches, the runtime stage
# carries neither. Non-root, and designed to run with a read-only root
# filesystem — a service whose whole point is access control should not run as
# root, and the process holds an index that is a denormalised copy of who may
# read what, so the blast radius of a compromise is the corpus. Give it as
# little as it can work with.
#
# What is deliberately absent from the runtime stage:
#
#   * a compiler and the pip cache — they exist in the build stage only,
#   * curl and any other HTTP client — the healthcheck uses the interpreter that
#     is already here, so the image has one fewer binary for a foothold,
#   * the test suite, the evaluation harness, and the docs.
#
# Pinned to 3.12 because that is what CI's matrix covers.

FROM python:3.12-slim-bookworm AS build

# Which extras to install. The default is the serving path; docker-compose.yml
# overrides it to add the Postgres backend. The in-process backends need nothing
# at all, and pulling ~1 GB of wheels to run a demo is a cost nobody agreed to.
ARG EXTRAS=".[qdrant,obs]"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build

# Dependency metadata first, so a source-only change does not re-resolve wheels.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install "${EXTRAS}"

FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="sightline" \
      org.opencontainers.image.description="Internal AI search that can only see what you can see." \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The only paths the process may write. Both are tmpfs mounts in
    # docker-compose.yml; running this image by hand with --read-only means
    # mounting them yourself. Pointing HOME at /tmp rather than at a home
    # directory keeps `--read-only` from turning a library's cache write into a
    # startup crash.
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    SIGHTLINE_ENV=production
WORKDIR /app

# A dedicated unprivileged user, created before the copy so the layer is cached.
# No home directory: nothing in the runtime path writes to one, and a writable
# directory that nothing writes to is a directory an attacker writes to.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin sightline
COPY --from=build /opt/venv /opt/venv
COPY --chown=sightline:sightline examples ./examples

USER sightline
EXPOSE 8000

# readyz, not healthz: healthz answers "is the process alive", readyz answers
# "can this instance serve a permission-checked query" — it reads the policy
# epoch — and only the second one should control whether traffic arrives. A
# server that cannot read the policy must choose between refusing and guessing,
# and only one of those is safe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status==200 else 1)"

# No --reload and no --workers: one process per container, scaled by the
# orchestrator. Workers here would give each its own plan cache and make "which
# epoch served this request" a per-worker question.
CMD ["uvicorn", "sightline.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-server-header", "--proxy-headers"]
