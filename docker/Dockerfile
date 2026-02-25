# Use a lightweight Python base image
FROM python:3.12-slim

# Set environment variables to configure Poetry and Python
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_NO_ANSI=1

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy the dependency files first to leverage Docker's layer caching
# The asterisk on poetry.lock* ensures the build doesn't fail if you haven't generated a lockfile yet
COPY pyproject.toml poetry.lock* ./

# Install dependencies
# --only main prevents installing development dependencies (like pytest) in the production image
# --no-root skips installing the project itself as a package (standard for simple Streamlit apps)
RUN poetry install --only main --no-root

# Copy the rest of your application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "ui/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
