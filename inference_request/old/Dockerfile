FROM python:3.11-slim
# 注意需要在 linux amd64 环境下编译

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install tritonclient[grpc]
RUN pip install scikit-learn

COPY . .
#ENTRYPOINT ["python3", "perf_analyzer.py"]




