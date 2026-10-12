FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY briefing ./briefing
RUN uv sync --frozen --no-dev --no-editable

RUN useradd --uid 1000 --create-home app && mkdir -p /data/runs && chown -R app /data
USER app

# Cloud Run sets PORT; STORE_URL is gs://bucket/prefix there.
ENV PATH=/opt/venv/bin:$PATH \
    PORT=8080 \
    STORE_URL=file:///data/runs
EXPOSE 8080
CMD ["sh", "-c", "exec streamlit run /app/briefing/ui.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
