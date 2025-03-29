FROM python:3.9-slim

WORKDIR /app

# システムパッケージのインストール例
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリのインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY ./app /app/app

# FastAPIやPrometheus用に必要なポートをEXPOSE
EXPOSE 8080

# メインアプリを実行
CMD ["python", "-m", "app.main"]