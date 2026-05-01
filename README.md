# 🤖 Universal ML Deploy Pipeline

**One click. Any model. Any hardware. Optimized and ready to deploy.**

## 🚀 Quick Start

1. Go to **Actions** tab
2. Click **"Universal ML Deploy Pipeline"**
3. Click **"Run workflow"**
4. Pick your options:
   - **Model**: Choose from dropdown or enter HuggingFace ID
   - **Hardware**: CPU, Raspberry Pi, NVIDIA GPU, Mobile, Edge TPU
   - **Priority**: Balanced, Speed, Size, or Accuracy
5. Click **Run**
6. Wait 3-5 minutes
7. Download `deploy-package` artifact — ready to use!

## 📦 What You Get

| Output | Description |
|--------|-------------|
| `model.onnx` | Optimized model for your hardware |
| `infer.py` | Ready-to-run inference script |
| `README.md` | Deployment instructions |
| `requirements.txt` | Dependencies |

## 🎯 Supported Models

| Source | Examples |
|--------|----------|
| **Torchvision** | resnet18, resnet50, mobilenet_v2, efficientnet_b0 |
| **HuggingFace** | Roboflow/rf-detr-base, facebook/detr-resnet-50, any HF model |
| **URL** | Direct link to .onnx, .pt, or .tflite file |

## 🖥️ Supported Hardware

| Target | Best For | Typical Winner |
|--------|----------|--------------|
| **CPU** | Servers, laptops | INT8 or FP16 |
| **Raspberry Pi** | Edge devices, IoT | INT8 (small, fast) |
| **NVIDIA GPU** | Cloud inference | FP16 (TensorRT-ready) |
| **Mobile Android** | Phones, tablets | INT8 → TFLite |
| **Mobile iOS** | iPhone, iPad | FP16 → CoreML |
| **Edge TPU** | Coral, Google devices | INT8 (compiled) |

## ⚙️ Optimization Techniques

| Technique | Size | Speed | Accuracy | Best For |
|-----------|------|-------|----------|----------|
| **FP16** | 50% smaller | 1.5x faster | ~0.5% drop | GPU, iOS |
| **INT8** | 75% smaller | 2-3x faster | ~1-2% drop | CPU, Edge, Android |
| **ORT Optimization** | Same | 1.2x faster | 0% drop | All targets |

## 📊 Example: RF-DETR on Raspberry Pi
