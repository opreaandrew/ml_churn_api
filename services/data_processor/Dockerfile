FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Using local files relative to build context (services/data_processor)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY processor ./processor
CMD ["python","-m","processor.run"]
