FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY site/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY site/ ./
RUN useradd --create-home --uid 1000 tynys && chown -R tynys:tynys /app
USER tynys
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
