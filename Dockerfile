FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only packaging metadata first for better layer caching
COPY pyproject.toml README.md /app/
COPY sufes /app/sufes

# Install the package (and its dependencies) in the image
RUN python -m pip install --upgrade pip \
    && python -m pip install .

# Default to the package CLI; pass args at `docker run` time.
ENTRYPOINT ["python", "-m", "sufes"]
CMD ["--help"]
