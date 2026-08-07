# FROM public.ecr.aws/deep-learning-containers/pytorch:2.13-cpu-amzn2023-sagemaker

# WORKDIR /opt/program

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY serve.py .
# COPY inference_utils.py .
# COPY transformer ./transformer

# ENV PYTHONBUFFERED=1
# ENV MODEL_DIR=/opt/ml/model

# EXPOSE 8080

# ENTRYPOINT ["python", "serve.py"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/opt/ml/model

WORKDIR /opt/program

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY inference.py .
COPY model_loader.py .
COPY serve.py .
COPY transformer ./transformer

EXPOSE 8080

ENTRYPOINT ["python", "serve.py"]