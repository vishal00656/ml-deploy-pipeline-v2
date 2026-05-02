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
    infer_script = (
        "#!/usr/bin/env python3\n"
        "# Ready-to-run inference script\n"
        "# Hardware: " + HARDWARE + "\n"
        "# Model: " + MODEL_ID + "\n"
        "\n"
        "import onnxruntime as rt\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "\n"
        "def load_model(model_path='model.onnx'):\n"
        "    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])\n"
        "    return sess\n"
        "\n"
        "def preprocess(image_path, input_size=224):\n"
        "    img = Image.open(image_path).convert('RGB')\n"
        "    img = img.resize((input_size, input_size))\n"
        "    arr = np.array(img).astype(np.float32) / 255.0\n"
        "    mean = np.array([0.485, 0.456, 0.406])\n"
        "    std = np.array([0.229, 0.224, 0.225])\n"
        "    arr = (arr - mean) / std\n"
        "    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]\n"
        "\n"
        "def predict(sess, image_path):\n"
        "    inp = sess.get_inputs()[0]\n"
        "    input_shape = [d if isinstance(d, int) and d > 0 else 224 for d in inp.shape]\n"
        "    input_size = input_shape[-1] if len(input_shape) >= 3 else 224\n"
        "    input_tensor = preprocess(image_path, input_size).astype(np.float32)\n"
        "    outputs = sess.run(None, {inp.name: input_tensor})\n"
        "    return outputs\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    import sys\n"
        "    if len(sys.argv) < 2:\n"
        "        print('Usage: python infer.py image.jpg')\n"
        "        sys.exit(1)\n"
        "    print('Loading model...')\n"
        "    sess = load_model()\n"
        "    print('Running inference on', sys.argv[1], '...')\n"
        "    outputs = predict(sess, sys.argv[1])\n"
        "    print('Output shapes:', [o.shape for o in outputs])\n"
        "    print('Done!')\n"
    )
    
    with open('deploy/infer.py', 'w') as f:
        f.write(infer_script)
    
    # Create README
    readme = (
        "# Deploy Package\n"
        "\n"
        "**Model:** " + MODEL_ID + "\n"
        "**Hardware:** " + HARDWARE + "\n"
        "**Format:** ONNX\n"
        "\n"
        "## Quick Start\n"
        "\n"
        "```bash\n"
        "pip install onnxruntime numpy Pillow\n"
        "python infer.py your_image.jpg\n"
        "```\n"
        "\n"
        "## Files\n"
        "\n"
        "- `model.onnx` — Optimized model\n"
        "- `infer.py` — Inference script\n"
    )
    
    with open('deploy/README.md', 'w') as f:
        f.write(readme)
    
    # Create requirements
    with open('deploy/requirements.txt', 'w') as f:
        f.write('onnxruntime\nnumpy\nPillow\n')
    
    # Create zip
    zip_path = 'deploy/' + MODEL_ID.replace('/', '_') + '_' + HARDWARE + '.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in ['model.onnx', 'infer.py', 'README.md', 'requirements.txt']:
            zf.write('deploy/' + file, file)
    
    print('✅ Deploy package created: ' + zip_path)

if __name__ == '__main__':
    create_deploy_package()
