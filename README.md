# AI Model Container Runtime Profiling (AC-Prof) Dataset
> Reproducible measurements of the invocation latency of AI Services Docker Containers, including cold starts and runtime behavior, under various resource specifications and input scales.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#continuous-integration) [![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#requirements) [![Issues](https://img.shields.io/github/issues/<ORG_OR_USER>/<REPO>.svg)](https://github.com/<ORG_OR_USER>/<REPO>/issues)

## Overview
This repository provides 
1. A dataset of latency measurements for popular AI service containers (with deep models at the core)
2. Scripts for systematically profiling containerized ML workloads. 
We focus on two critical aspects for scheduling and resource management: container cold start and nonlinear runtime behavior under varid CPU, GPU, and memory specs as well as input sizes. The dataset is designed to support reproducible performance modeling and quantitative evaluation of scheduling and resource allocation strategies.

## What’s Included
- **Static metrics**: container image and model weights size (download volume).
- **Dynamic metrics**: end-to-end runtime measured under a specified matrix of CPU/GPU/memory limits and input scales.
- **Optional metrics**: peak CPU usage, peak GPU utilization or VRAM usage.
- **Scripts**: Docker build/run recipes with resource caps, client scripts for warm-up and request timing, CSV logging, and plotting utilities.

## Measurement Environment
- OS: Ubuntu 24.04.6 LTS  
- Container runtime: Docker 27.5.1  
- Drivers/Libraries: CUDA 12.1, cuDNN 9.1  
- Language/Framework: Python 3.12, PyTorch 2.5.1+cu121

## Data Sources
This dataset is collected with reference to the **APIBench** dataset methodology. External model APIs are sourced from three popular ML model repositories:
- **TorchHub**: https://pytorch.org/hub/
- **TensorFlow Hub**: https://www.tensorflow.org/hub
- **HuggingFace Models**: https://huggingface.co/models

## Resource and Input Matrix
- CPU cores: `{1, 2, 4, 8}`  
- Memory limits: `{2 GB, 4 GB, 8 GB, 16 GB}`  
- GPU count: `{0, 1}`  
- Input scaling: task-specific multi-level inputs (e.g., image resolution or text length grid)

## Data Example
![runtime profile example](docs/container_runtime_example.png)

# Guideline for Customized Profiling

## Requirements
- Python 3.10 or newer
- Git and a recent C or C++ toolchain if native dependencies are required
- Optional CUDA 12.x for GPU features

## What to Record
- Startup metrics: image download size and time, container initialization time,.
- Runtime metrics: per-request end-to-end latency under each CPU/MEM/GPU/input configuration.
- Optional peaks: peak CPU usage, peak GPU utilization/VRAM.

## Quick Start Tutorial: profiling FCN-50
#### 1) Model Download (download_model.py)
```bash
import os
import torch
import torchvision
import shutil

# Set local storage directory (ensure models are downloaded here)
TORCH_HOME = "data/torch_hub"
os.makedirs(TORCH_HOME, exist_ok=True)  # Ensure the directory exists

# Only effective within the current process, does not affect the global `TORCH_HOME`
torch.hub.set_dir(TORCH_HOME)

# Model information
MODEL_REPO = "pytorch/vision"
MODEL_NAME = "fcn_resnet50"
PRETRAINED = True

print(f"Pre-downloading model: {MODEL_NAME}, repository: {MODEL_REPO}, storage path: {TORCH_HOME}")
model = torch.hub.load(repo_or_dir=MODEL_REPO, model=MODEL_NAME, pretrained=PRETRAINED)
print(f"Model pre-download complete, storage path: {TORCH_HOME}")
```

#### 2) Write inference service code（server.py）
```bash
import torch
import numpy as np
import traceback
from flask import Flask, request, jsonify
import os
import torchvision.models as models

app = Flask(__name__)

# ===================== 🔥 加载 FCN-ResNet50 模型 =====================
print("🚀 正在加载 FCN-ResNet50 模型...")
os.environ["TORCH_HOME"] = "/opt/torch_hub/"
torch.hub.set_dir("/opt/torch_hub/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 FCN-ResNet50 分割模型
model = models.segmentation.fcn_resnet50(pretrained=True)
model.to(device).eval()
print("✅ 模型加载完成！")

# ===================== 🔥 API 接口 =====================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📥 收到张量处理请求...")

        # 解析 JSON 数据
        data = request.get_json()
        if data is None or "tensor" not in data:
            print("❌ 请求数据格式错误")
            return jsonify({"error": "请提供 'tensor' 字段，表示输入图像的张量"}), 400

        # 将 JSON 中的 'tensor' 转换为 NumPy 数组，并转为 torch.Tensor
        try:
            input_array = np.array(data["tensor"], dtype=np.float32)
            input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        except Exception as e:
            print("❌ 'tensor' 解析失败:", str(e))
            return jsonify({"error": "无效的 'tensor' 格式"}), 400

        # 检查输入张量形状是否为 (1, 3, H, W)，H 和 W 可变
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1 or input_tensor.shape[1] != 3:
            print(f"❌ 输入张量形状错误: {input_tensor.shape}")
            return jsonify({"error": "输入张量形状错误，期望形状为 (1, 3, H, W)"}), 400

        print(f"✅ 接收张量，形状: {input_tensor.shape}")

        # 进行模型推理
        with torch.no_grad():
            # 模型返回一个字典，"out" 是分割结果
            output = model(input_tensor)["out"][0]

        # 生成分割结果（取每个像素的类别索引）
        output_predictions = output.argmax(0).cpu().numpy()
        print("✅ 预测完成，返回结果！")
        return jsonify({"segmentation": output_predictions.tolist()})
    
    except Exception as e:
        print("❌ 服务器处理异常:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ===================== 🚀 启动 Flask 服务器 =====================
if __name__ == '__main__':
    print("🌍 服务器启动中...")
    app.run(host="0.0.0.0", port=8006, threaded=True)

```
#### 3) Compose the Dockerfile
```bash
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install --default-timeout=600 torch torchvision pytorchvideo flask opencv-python-headless flask -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV TORCH_HOME=/opt/torch_hub

WORKDIR /opt

COPY download_model.py server.py /opt/

COPY data/torch_hub/ /opt/torch_hub/

EXPOSE 8006

CMD ["python", "server.py"]
```

#### 4) Deploy containers with specified resource budgets
```bash
sudo docker run --gpus all -d -p 9001:8006 --cpus="8.0" --memory="16g" --name fcn_resnet50_gpu torchhub_fcn_resnet50_server
sudo docker run -d -p 9002:8006 --cpus="6.0" --memory="12g" --name fcn50_cpu6 torchhub_fcn_resnet50_server
sudo docker run -d -p 9003:8006 --cpus="4.0" --memory="8g" --name fcn50_cpu4 torchhub_fcn_resnet50_server
sudo docker run -d -p 9004:8006 --cpus="3.0" --memory="6g" --name fcn50_cpu3 torchhub_fcn_resnet50_server
sudo docker run -d -p 9005:8006 --cpus="2.0" --memory="4g" --name fcn50_cpu2 torchhub_fcn_resnet50_server
sudo docker run -d -p 9006:8006 --cpus="1.0" --memory="2g" --name fcn50_cpu1 torchhub_fcn_resnet50_server
```

#### 5）Metrics collection code (setting different input images can collect runtime metrics given different input sizes)
```bash
import requests
import numpy as np
import cv2
import time
import json
import csv
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# ---------------------------
# Load local image and convert to RGB
# ---------------------------
def load_local_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Failed to read image, please check the path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ---------------------------
# Define the preprocessing pipeline (based on official sample code)
# ---------------------------
# Use torchvision transforms to convert the image to a tensor and normalize it
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Test function: Apply preprocessing transform to the given image and call the API to test inference time
# ---------------------------
def test_inference(api_url, img, orig_size):
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)
    # Apply the preprocessing pipeline to get the tensor
    input_tensor = preprocess(img_pil)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    payload = {
        "tensor": input_tensor.cpu().numpy().tolist(),
        "orig_size": orig_size  # Original image size [height, width]
    }
    
    start = time.perf_counter()
    response = requests.post(api_url, json=payload, timeout=60)
    elapsed = time.perf_counter() - start
    return response, elapsed

# ---------------------------
# Main function
# ---------------------------
def main():
    # Local image path (please modify according to your setup)
    image_path = "/home/lishenghai/桌面/tool_set/MiDas_Hybrid/flower.jpg"
    original_img = load_local_image(image_path)
    
    # Original image size [height, width]
    orig_size = [original_img.shape[0], original_img.shape[1]]
    
    # API service address (please modify based on your deployment)
    api_url = "http://localhost:8006/predict"
    
    # ---------------------------
    # Warm-up step: Use the original image for one warm-up call
    # ---------------------------
    print("Warming up the container...")
    _, warmup_time = test_inference(api_url, original_img, orig_size)
    print(f"Warm-up call took: {warmup_time:.4f} seconds\n")
    
    # Define 20 scaling factors, from 0.1 to 2.0
    scales = np.linspace(0.1, 2.0, 20)
    results = []
    
    print("Starting test on the impact of different scaled image sizes on total inference time...")
    for s in scales:
        new_width = int(original_img.shape[1] * s)
        new_height = int(original_img.shape[0] * s)
        img_scaled = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # For the server, orig_size remains the original image size
        response, elapsed = test_inference(api_url, img_scaled, orig_size)
        print(f"Scale factor {s:.2f}, resized to {new_height}x{new_width}, request time {elapsed:.4f} seconds")
        results.append({
            "scale": s,
            "width": new_width,
            "height": new_height,
            "time": elapsed
        })
    
    print("\nTest Results:")
    for res in results:
        print(res)
    
    # Save test results to CSV file (keep four decimal places)
    with open("inference_time_results.csv", "w", newline="") as csvfile:
        fieldnames = ["scale", "width", "height", "time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({
                "scale": f"{res['scale']:.4f}",
                "width": res["width"],
                "height": res["height"],
                "time": f"{res['time']:.4f}"
            })
    print("Test results saved to inference_time_results.csv")
    
    # Plot the line chart
    scales_plot = [res["scale"] for res in results]
    times_plot = [res["time"] for res in results]
    
    plt.figure(figsize=(8, 5))
    plt.plot(scales_plot, times_plot, '-o', label='Inference Time')
    plt.xlabel("Scale Factor")
    plt.ylabel("Time (seconds)")
    plt.title("Inference Time vs Scale Factor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
```

## Modeling Guidance
- After collecting measurements for a container, fit a simple parametric or piecewise model (e.g., least squares) for latency as a function of resources and input size, and report goodness of fit and residuals. Keep train/test splits separate for each container–task pair.


## Contribution
New contributors are welcome. Please open an issue to discuss your idea before submitting a pull request. Follow the code style and ensure tests pass. See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` if present.

## License
This project is released under the Apache-2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements
This dataset is part of the DOR project (https://github.com/wingter562/DISTINT_open_data) by Dr. Wentai Wu, Jinan University, with primary contribution by Dr. Shenghai Li, South China University of Technology.

**List of contributors:**
- Wentai Wu, JNU
- Shenghai Li, SCUT
- Kaizhe Song, JNU
- Qinan Wu, JNU
- Yukai Wang, JNU

Project contact: wentaiwu[at]jnu[dot]edu[dot]cn | lishenghai2022[at]foxmail[dot]com

Issues and feature requests: please open a GitHub Issue
