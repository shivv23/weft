# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# uv for reproducible, fast installs from the lockfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Non-root runtime
RUN useradd --create-home --uid 10001 weft \
    && chown -R weft:weft /app
USER weft

EXPOSE 8080

# Default entrypoint runs the REST API; override with `weft demo`, `weft bench`, ...
ENTRYPOINT ["weft"]
CMD ["api"]
