# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy only the midterm project files
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the midterm directory contents
COPY --chown=user midterm /app

# Set environment variable for Python path
ENV PYTHONPATH=/app/src

# Run streamlit on port 7860 for Hugging Face Spaces
CMD ["streamlit", "run", "src/ui/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
