import os
import shutil
import zipfile

HARDWARE = os.environ.get('HARDWARE_TARGET', 'cpu')
MODEL_ID = os.environ.get('MODEL_ID', 'model')

def create_deploy_package():
    """Create ready-to-deploy package"""
    os.makedirs('deploy', exist_ok=True)
    
    # Copy winner model
    shutil.copy('output/model_winner.onnx', 'deploy/model.onnx')
    
    # Create inference script
    infer_script = f'''#!/usr/bin/env python3
"""
Ready-to-run inference script
Hardware: {HARDWARE}
Model: {MODEL_ID}
"""

import onnxruntime as rt
import numpy as np
from PIL import Image

def load_model(model_path='model.onnx'):
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return sess

def preprocess(image_path, input_size=224):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_size, input_size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

def predict(sess, image_path):
    inp = sess.get_inputs()[0]
    input_shape = [d if isinstance(d, int) and d > 0 else 224 for d in inp.shape]
    input_size = input_shape[-1] if len(input_shape) >= 3 else 224
    
    input_tensor = preprocess(image_path, input_size).astype(np.float32)
    outputs = sess.run(None, {inp.name: input_tensor})
    return outputs

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python infer.py image.jpg')
        sys.exit(1)
    
    print('Loading model...')
    sess = load_model()
    
    print(f'Running inference on {{sys.argv[1]}}...')
    outputs = predict(sess, sys.argv[1])
    
    print(f'Output shapes: {{[o.shape for o in outputs]}}')
    print('Done!')
'''
    
    with open('deploy/infer.py', 'w') as f:
        f.write(infer_script)
    
    # Create README
    readme = f'''# Deploy Package

**Model:** {MODEL_ID}
**Hardware:** {HARDWARE}
**Format:** ONNX

## Quick Start

```bash
pip install onnxruntime numpy Pillow
python infer.py your_image.jpg