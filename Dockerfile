# Use a pre-configured image with uv and python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Install dependencies only (cached layer)
# --frozen: strict usage of lockfile
# --no-install-project: don't install the current project yet (avoids cache busting when code changes)
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .

# Install the project itself (if needed) and sync environment
RUN uv sync --frozen --no-dev

# Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck to ensure the app is responsive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
