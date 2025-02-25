# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the PDF file first
COPY --chown=user Grid_Code.pdf Grid_Code.pdf

# Copy the rest of the application
COPY --chown=user . .

# Set environment variables for Python path
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src"

# Run streamlit on port 7860 for Hugging Face Spaces
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]