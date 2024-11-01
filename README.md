# Yolo v8 と mediapipeを用いて被採点者の骨格抽出を行う

## 環境構築
### dockerコンテナの作成
```bash
$ docker pull huggingface/transformers-pytorch-gpu:latest
$ docker run -it \
    -w /app \
    -v .:/app \
    -p 8080:8080 \
    --shm-size 16g \
    --gpus all \
    --name med_yolo
    huggingface/transformers-pytorch-gpu:latest bash
```

### コンテナの中身を作成
```bash
$ apt-get update -y && apt update -y
$ apt install curl git
```

### ライブラリのインストール
```bash
$ pip install ultralytics
```
