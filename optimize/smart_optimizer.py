import onnx
import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import os
import time
import json
import sys

INPUT_MODEL = 'input/model.onnx'
OUTPUT_DIR = 'output'
RESULTS_FILE = 'output/results.json'

HARDWARE = os.environ.get('HARDWARE_TARGET', 'cpu')
PRIORITY = os.environ.get('PRIORITY', 'balanced').lower()

# Hardware profiles
PROFILES = {
    'cpu': {'max_size_mb': 200, 'max_infer_ms': 1000, 'max_acc_drop': 5.0, 'techniques': ['fp16', 'int8', 'ort_opt']},
    'raspberry_pi': {'max_size_mb': 50, 'max_infer_ms': 2000, 'max_acc_drop': 10.0, 'techniques': ['int8', 'ort_opt']},
    'nvidia_gpu': {'max_size_mb': 500, 'max_infer_ms': 100, 'max_acc_drop': 2.0, 'techniques': ['fp16', 'ort_opt']},
    'mobile_android': {'max_size_mb': 50, 'max_infer_ms': 500, 'max_acc_drop': 5.0, 'techniques': ['int8', 'ort_opt']},
    'mobile_ios': {'max_size_mb': 50, 'max_infer_ms': 500, 'max_acc_drop': 5.0, 'techniques': ['fp16', 'ort_opt']},
    'edge_tpu': {'max_size_mb': 8, 'max_infer_ms': 100, 'max_acc_drop': 10.0, 'techniques': ['int8']}
}

profile = PROFILES.get(HARDWARE, PROFILES['cpu'])
MAX_SIZE_MB = profile['max_size_mb']
MAX_INFER_MS = profile['max_infer_ms']
MAX_ACC_DROP = profile['max_acc_drop']
ALLOWED_TECHNIQUES = profile['techniques']

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def measure_inference_ms(model_path, runs=10):
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    dtype = np.float16 if inp.type == 'tensor(float16)' else np.float32
    dummy = np.random.randn(*shape).astype(dtype)
    
    for _ in range(2):
        sess.run(None, {inp.name: dummy})
    
    t0 = time.time()
    for _ in range(runs):
        sess.run(None, {inp.name: dummy})
    return (time.time() - t0) / runs * 1000

def estimate_accuracy_drop(original_path, optimized_path, samples=20):
    sess_orig = rt.InferenceSession(original_path, providers=['CPUExecutionProvider'])
    sess_opt = rt.InferenceSession(optimized_path, providers=['CPUExecutionProvider'])
    inp_orig = sess_orig.get_inputs()[0]
    inp_opt = sess_opt.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp_orig.shape]
    
    diffs = []
    for _ in range(samples):
        dummy = np.random.randn(*shape).astype(np.float32)
        out_o = sess_orig.run(None, {inp_orig.name: dummy})[0].flatten().astype(np.float32)
        out_q = sess_opt.run(None, {inp_opt.name: dummy})[0].flatten().astype(np.float32)
        denom = np.abs(out_o) + 1e-8
        diffs.append(np.mean(np.abs(out_o - out_q) / denom) * 100)
    return float(np.mean(diffs))

def passes_gates(size_mb, acc_drop, infer_ms):
    return size_mb <= MAX_SIZE_MB and infer_ms <= MAX_INFER_MS and acc_drop <= MAX_ACC_DROP

def technique_fp16(input_path, output_path):
    from onnxconverter_common import float16
    model = onnx.load(input_path)
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(fp16_model, output_path)

def technique_int8(input_path, output_path):
    quantize_dynamic(
        input_path, output_path,
        weight_type=QuantType.QInt8,
        optimize_model=False,
        extra_options={'MatMulConstBOnly': True}
    )

def technique_ort_opt(input_path, output_path):
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path
    _ = rt.InferenceSession(input_path, sess_options, providers=['CPUExecutionProvider'])

def run():
    print(f"\n📦 Model: {INPUT_MODEL}")
    print(f"🎯 Hardware: {HARDWARE.upper()}")
    print(f"🎚️ Priority: {PRIORITY.upper()}")
    
    original_mb = get_size_mb(INPUT_MODEL)
    print(f"📊 Original size: {original_mb:.2f} MB")
    print(f"⚙️ Allowed techniques: {ALLOWED_TECHNIQUES}\n")
    
    all_techniques = {
        'fp16': ('output/model_fp16.onnx', technique_fp16),
        'int8': ('output/model_int8.onnx', technique_int8),
        'ort_opt': ('output/model_ort_opt.onnx', technique_ort_opt)
    }
    
    results = []
    for name in ALLOWED_TECHNIQUES:
        if name not in all_techniques:
            continue
            
        out_path, fn = all_techniques[name]
        print(f"🔧 Running: {name}")
        
        try:
            fn(INPUT_MODEL, out_path)
            size_mb = get_size_mb(out_path)
            infer_ms = measure_inference_ms(out_path)
            acc_drop = estimate_accuracy_drop(INPUT_MODEL, out_path)
            gate_ok = passes_gates(size_mb, acc_drop, infer_ms)
            
            results.append({
                'technique': name,
                'size_mb': round(size_mb, 3),
                'infer_ms': round(infer_ms, 2),
                'acc_drop': round(acc_drop, 3),
                'gate_pass': gate_ok,
                'status': 'ok'
            })
            icon = '✅' if gate_ok else '⚠️'
            print(f"   {icon} size={size_mb:.2f}MB  infer={infer_ms:.1f}ms  acc_drop={acc_drop:.2f}%")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({'technique': name, 'status': 'failed', 'error': str(e)})
    
    passing = [r for r in results if r.get('status') == 'ok']
    if not passing:
        print("\n❌ All techniques failed!")
        sys.exit(1)
    
    # Pick winner based on priority
    gate_passing = [r for r in passing if r.get('gate_pass')]
    candidates = gate_passing if gate_passing else passing
    
    if PRIORITY == 'speed':
        winner = min(candidates, key=lambda r: r['infer_ms'])
    elif PRIORITY == 'size':
        winner = min(candidates, key=lambda r: r['size_mb'])
    elif PRIORITY == 'accuracy':
        winner = min(candidates, key=lambda r: r['acc_drop'])
    else:  # balanced - prefer smallest that passes
        winner = min(candidates, key=lambda r: r['size_mb'] + r['infer_ms']/100)
    
    print(f"\n🏆 WINNER: {winner['technique']}")
    print(f"   Size: {winner['size_mb']}MB | Speed: {winner['infer_ms']}ms | Acc drop: {winner['acc_drop']}%")
    
    import shutil
    shutil.copy(f"output/model_{winner['technique']}.onnx", 'output/model_winner.onnx')
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'original_mb': round(original_mb, 3),
            'winner': winner,
            'all_results': results,
            'hardware': HARDWARE,
            'priority': PRIORITY
        }, f, indent=2)
    
    print(f"\n💾 Results saved → {RESULTS_FILE}")

if __name__ == '__main__':
    run()