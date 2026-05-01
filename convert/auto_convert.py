import os
import sys
import torch
import onnx
from transformers import AutoModel, AutoModelForObjectDetection, AutoImageProcessor
import torchvision.models as models

MODEL_SOURCE = os.environ.get('MODEL_SOURCE', 'huggingface')
MODEL_ID = os.environ.get('MODEL_ID', 'Roboflow/rf-detr-base')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def convert_huggingface():
    """Convert HuggingFace model to ONNX"""
    print(f"Loading HuggingFace model: {MODEL_ID}")
    
    # Try object detection first
    try:
        model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model.eval()
        
        # Get input size from processor
        size = getattr(processor, 'size', {'height': 560, 'width': 560})
        if isinstance(size, dict):
            h = size.get('height', 560)
            w = size.get('width', 560)
        else:
            h = w = 560
        
        dummy_input = torch.randn(1, 3, h, w)
        
        ensure_dir('input')
        torch.onnx.export(
            model,
            dummy_input,
            'input/model.onnx',
            input_names=['pixel_values'],
            output_names=['logits', 'pred_boxes'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'pred_boxes': {0: 'batch_size'}
            },
            opset_version=14
        )
        print(f"✅ HF Object Detection model exported: {h}x{w}")
        return True
        
    except Exception as e:
        print(f"Not an object detection model or error: {e}")
    
    # Try generic model
    try:
        model = AutoModel.from_pretrained(MODEL_ID)
        model.eval()
        
        # Assume standard image input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        ensure_dir('input')
        torch.onnx.export(
            model,
            dummy_input,
            'input/model.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=14
        )
        print("✅ HF Generic model exported")
        return True
        
    except Exception as e:
        print(f"Failed to load as generic model: {e}")
        return False

def convert_torchvision():
    """Convert torchvision model to ONNX"""
    print(f"Loading torchvision model: {MODEL_ID}")
    
    model_fn = getattr(models, MODEL_ID.replace('-', '_'), None)
    if model_fn is None:
        print(f"❌ Unknown torchvision model: {MODEL_ID}")
        print(f"Available: resnet18, resnet50, mobilenet_v2, efficientnet_b0, etc.")
        return False
    
    model = model_fn(weights='DEFAULT')
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    ensure_dir('input')
    torch.onnx.export(
        model,
        dummy_input,
        'input/model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    print("✅ Torchvision model exported")
    return True

def convert_from_url():
    """Download model from URL and convert"""
    import requests
    url = MODEL_ID
    
    print(f"Downloading model from: {url}")
    r = requests.get(url)
    
    ensure_dir('input')
    temp_path = 'input/downloaded_model'
    with open(temp_path, 'wb') as f:
        f.write(r.content)
    
    # Detect format and convert
    from convert.detect_format import detect_format, get_model_info
    fmt = detect_format(temp_path)
    info = get_model_info(fmt)
    
    print(f"Detected format: {fmt} - {info['description']}")
    
    if fmt == 'onnx':
        # Already ONNX, just move
        os.rename(temp_path, 'input/model.onnx')
        return True
    else:
        print(f"❌ URL format '{fmt}' not yet supported for auto-convert")
        return False

def main():
    ensure_dir('input')
    
    if MODEL_SOURCE == 'huggingface':
        success = convert_huggingface()
    elif MODEL_SOURCE == 'torchvision':
        success = convert_torchvision()
    elif MODEL_SOURCE == 'url':
        success = convert_from_url()
    else:
        print(f"❌ Unknown source: {MODEL_SOURCE}")
        success = False
    
    if not success:
        sys.exit(1)
    
    # Validate the ONNX
    model = onnx.load('input/model.onnx')
    onnx.checker.check_model(model)
    print("✅ ONNX validation passed")

if __name__ == '__main__':
    main()