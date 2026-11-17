FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CPU=1

WORKDIR /srv

# 先装 CPU 版 PyTorch（独立 index，镜像内不引入 CUDA 依赖）
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

# 再装应用依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY static ./static

EXPOSE 8000
# 容器内线程数收敛，CPU 环境行为稳定
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
