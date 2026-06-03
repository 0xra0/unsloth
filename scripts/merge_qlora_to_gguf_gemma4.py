#!/usr/bin/env python3
"""
QLoRA merge + GGUF export for gemma-4-e4b-it.

Strategy: replace each Linear4bit in-place on GPU, move to CPU immediately.
Peak VRAM stays at ~11 GB instead of the full-model ~16 GB that merge_and_unload needs.
"""
import sys
sys.path.insert(0, '/home/home/.unsloth/studio/.venv_t5_550')

import os, gc, json, shutil, subprocess
import torch
import bitsandbytes as bnb
from safetensors import safe_open
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL   = '/home/home/.cache/huggingface/hub/models--unsloth--gemma-4-e4b-it-unsloth-bnb-4bit/snapshots/91872ae26ac6bfbae6ee6a8acdc73d3370d1d227'
LORA_ADAPTER = '/home/home/.unsloth/studio/outputs/unsloth_gemma-4-E4B-it_1780058619'
OUTPUT_BF16  = '/mnt/sata/tmp_gemma4_bf16'
OUTPUT_GGUF  = '/home/home/.unsloth/studio/exports/gemma-4-e4b-it-unsloth-bnb-4bit-gguf'
MODEL_NAME   = 'gemma-4-e4b-it'
CONVERTER    = '/home/home/.unsloth/llama.cpp/convert_hf_to_gguf.py'
QUANTIZER    = '/home/home/.unsloth/llama.cpp/build/bin/llama-quantize'
GGUF_PY      = '/home/home/.unsloth/llama.cpp/gguf-py'

shutil.rmtree(OUTPUT_BF16, ignore_errors=True)
os.makedirs(OUTPUT_BF16)
os.makedirs(OUTPUT_GGUF, exist_ok=True)

# ── Step 1: Load LoRA weights (CPU) ─────────────────────────────────────────
print("Step 1: Loading LoRA weights...")
lora_weights = {}
with safe_open(LORA_ADAPTER + '/adapter_model.safetensors', framework='pt', device='cpu') as f:
    for k in f.keys():
        lora_weights[k] = f.get_tensor(k)

with open(LORA_ADAPTER + '/adapter_config.json') as f:
    adapter_cfg = json.load(f)
LORA_SCALE = adapter_cfg['lora_alpha'] / adapter_cfg['r']
print(f"  {len(lora_weights)} LoRA tensors, scale={LORA_SCALE}")

def get_lora_keys(module_path):
    """Return (lora_A key, lora_B key) for a given module path in the base model."""
    prefix = f'base_model.model.{module_path}'
    return prefix + '.lora_A.weight', prefix + '.lora_B.weight'

# ── Step 2: Load base model (BNB 4-bit on GPU) ──────────────────────────────
print("Step 2: Loading BNB 4-bit base model on GPU...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map='cuda',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
print(f"  Loaded. GPU={torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Step 3: In-place merge: replace each Linear4bit on GPU, move result to CPU ──
print("Step 3: Replacing Linear4bit layers with merged bfloat16 Linear (GPU→CPU per layer)...")

n_merged = 0
n_passthrough = 0

# Collect (name, module) pairs first to avoid modifying dict during iteration
linear4bit_layers = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, bnb.nn.Linear4bit)
]
print(f"  Found {len(linear4bit_layers)} Linear4bit layers to process")

for name, module in linear4bit_layers:
    # Dequantize on GPU — quant_state lives on the Params4bit or on the module itself
    with torch.no_grad():
        qs = None
        if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
            qs = module.weight.quant_state
        elif hasattr(module, 'quant_state') and module.quant_state is not None:
            qs = module.quant_state

        if qs is not None:
            w = bnb.functional.dequantize_4bit(module.weight.data, qs).to(torch.bfloat16)
        elif module.weight.dtype in (torch.float16, torch.bfloat16):
            # BNB wrapped a float layer (e.g. audio tower) — use weight directly
            w = module.weight.data.to(torch.bfloat16)
        else:
            print(f"  SKIP {name}: dtype={module.weight.dtype} weight_type={type(module.weight).__name__} no quant_state")
            continue

    # Apply LoRA delta if this layer has an adapter
    lA_key, lB_key = get_lora_keys(name)
    if lA_key in lora_weights:
        lA = lora_weights[lA_key].to(w.device, dtype=torch.bfloat16)
        lB = lora_weights[lB_key].to(w.device, dtype=torch.bfloat16)
        w = w + (lB @ lA) * LORA_SCALE
        del lA, lB
        n_merged += 1
    else:
        n_passthrough += 1

    # Create a regular nn.Linear on CPU with the merged weight
    new_lin = torch.nn.Linear(
        module.in_features, module.out_features,
        bias=module.bias is not None,
        device='cpu',
        dtype=torch.bfloat16,
    )
    new_lin.weight = torch.nn.Parameter(w.cpu())
    if module.bias is not None:
        new_lin.bias = torch.nn.Parameter(module.bias.data.cpu().to(torch.bfloat16))

    # Replace the module in the model tree
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_lin)

    del w, module, new_lin
    torch.cuda.empty_cache()

print(f"  Merged with LoRA: {n_merged}, passthrough (dequant only): {n_passthrough}")
print(f"  GPU after linear replacement: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Step 3b: Replace remaining Linear4bit (vision/audio tower) with zero bfloat16 ──
# Vision/audio tower uses custom INT8 quant (not NF4) — they go to mmproj, not LM GGUF.
# We need plain nn.Linear so save_pretrained can serialize without crashing.
remaining_l4 = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, bnb.nn.Linear4bit)
]
print(f"  Replacing {len(remaining_l4)} remaining Linear4bit (vision/audio) with zero bfloat16 Linear...")
for name, module in remaining_l4:
    new_lin = torch.nn.Linear(
        module.in_features, module.out_features,
        bias=module.bias is not None,
        device='cpu', dtype=torch.bfloat16,
    )
    if module.bias is not None:
        new_lin.bias = torch.nn.Parameter(torch.zeros(module.out_features, dtype=torch.bfloat16))
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_lin)
print(f"  Done. GPU={torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Step 4: Move remaining GPU tensors to CPU ────────────────────────────────
print("Step 4: Moving remaining parameters to CPU...")
model = model.cpu()
torch.cuda.empty_cache()
print(f"  GPU after model.cpu(): {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Step 5: Save as bfloat16 safetensors ────────────────────────────────────
# Patch: Bnb4bitDeserialize lacks reverse_op so revert_weight_conversion crashes.
# We replaced every BNB module with plain bfloat16 nn.Linear — nothing to revert.
import transformers.modeling_utils as _mu
import transformers.core_model_loading as _cml
_cml.revert_weight_conversion = lambda model, sd: sd
_mu.revert_weight_conversion = lambda model, sd: sd

print(f"Step 5: Saving merged bfloat16 model to {OUTPUT_BF16} ...")
model.save_pretrained(
    OUTPUT_BF16,
    safe_serialization=True,
    max_shard_size='4GB',
)
print("  Model saved.")

del model
gc.collect()

# Copy tokenizer files + rewrite config.json without BNB quantization config
print("  Copying tokenizer files and cleaning config.json...")
for fname in os.listdir(BASE_MODEL):
    if not fname.endswith(('.json', '.jinja', '.model')):
        continue
    src = os.path.join(BASE_MODEL, fname)
    dst = os.path.join(OUTPUT_BF16, fname)
    if fname == 'config.json':
        with open(src) as f:
            cfg = json.load(f)
        cfg.pop('quantization_config', None)
        cfg['torch_dtype'] = 'bfloat16'
        with open(dst, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"  Rewrote config.json (quantization_config stripped)")
    elif not os.path.exists(dst):
        shutil.copy(src, dst)

# ── Step 6: Convert to GGUF BF16 ────────────────────────────────────────────
print("Step 6: Converting to GGUF BF16 ...")
gguf_bf16 = os.path.join(OUTPUT_GGUF, f'{MODEL_NAME}.BF16.gguf')

env = os.environ.copy()
env['PYTHONPATH'] = GGUF_PY + ':' + sys.path[0]

result = subprocess.run(
    [sys.executable, CONVERTER,
     OUTPUT_BF16,
     '--outtype', 'bf16',
     '--outfile', gguf_bf16],
    env=env,
)
if result.returncode != 0:
    print(f"ERROR: converter exited {result.returncode}")
    sys.exit(1)
print(f"  {gguf_bf16}: {os.path.getsize(gguf_bf16)/1e9:.2f} GB")

# ── Step 7: Quantize to Q4_K_M ─────────────────────────────────────────────
print("Step 7: Quantizing to Q4_K_M ...")
gguf_q4 = os.path.join(OUTPUT_GGUF, f'{MODEL_NAME}.Q4_K_M.gguf')

lib_dir = '/home/home/.unsloth/llama.cpp/build/bin'
env2 = os.environ.copy()
env2['LD_LIBRARY_PATH'] = lib_dir + (':' + env2.get('LD_LIBRARY_PATH', '') if env2.get('LD_LIBRARY_PATH') else '')

result = subprocess.run(
    [QUANTIZER, gguf_bf16, gguf_q4, 'Q4_K_M', str(os.cpu_count() or 8)],
    env=env2,
)
if result.returncode != 0:
    print(f"ERROR: quantizer exited {result.returncode}")
    sys.exit(1)
print(f"  {gguf_q4}: {os.path.getsize(gguf_q4)/1e9:.2f} GB")

# ── Step 8: Verify ───────────────────────────────────────────────────────────
print("Step 8: Verifying first 20 tensors non-zero...")
sys.path.insert(0, GGUF_PY)
from gguf import GGUFReader
import numpy as np

reader = GGUFReader(gguf_q4)
sample = reader.tensors[:20]
bad = [t.name for t in sample if np.all(t.data == 0)]
print(f"  Checked {len(sample)}: {len(bad)} all-zero tensors (should be 0)")
if bad:
    print("  BAD tensors:", bad)

print("\nDone! Rebuilt:", gguf_q4)
