# Start with Ollama as the base
FROM ollama/ollama:latest

# 1. Install Python and venv (essential for modern Debian)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# 2. Install Poetry using the official method to avoid "externally managed" errors
RUN python3 -m venv /opt/poetry-venv && \
    /opt/poetry-venv/bin/pip install poetry==$POETRY_VERSION

# Add Poetry to PATH
ENV PATH="/opt/poetry-venv/bin:$PATH"

# 3. Copy dependencies and install
# We let Poetry create its own virtual env in /app/.venv
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

# 4. Copy your app code
COPY . .

# 5. Create the startup script
# Note: We must run 'poetry run' so it uses the virtual environment we just created
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
poetry run streamlit run ui/Home.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8501
EXPOSE 11434

ENTRYPOINT ["/bin/bash", "/app/start.sh"]
