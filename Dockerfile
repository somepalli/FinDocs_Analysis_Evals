FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra api --extra docling --extra gpu --extra retrieval --extra observability \
    --no-install-project

COPY src ./src
COPY evals ./evals
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra api --extra docling --extra gpu --extra retrieval --extra observability

EXPOSE 8989

CMD ["findociq-api", "--host", "0.0.0.0", "--port", "8989", "--config", "configs/api/docker.yaml"]
