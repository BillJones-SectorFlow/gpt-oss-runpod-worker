#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, math
import torch
import pathlib, json, time
import argparse
import numpy as np
from torch.utils.cpp_extension import IS_HIP_EXTENSION
import subprocess
import shutil
import threading
from datetime import datetime

# Set tutel timeout BEFORE any tutel imports to ensure it takes effect
os.environ['TUTEL_GLOBAL_TIMEOUT_SEC'] = str(2147483647)
os.environ['D3D12_ENABLE_FP16'] = '1'

parser = argparse.ArgumentParser()
# Removed: --try_path
parser.add_argument('--hf_model', type=str, default='openai/gpt-oss-120b',
                    help='Hugging Face repo id to download if local model is missing (e.g. openai/gpt-oss-120b)')
parser.add_argument('--path_to_model', type=str, default='/data/models/gpt-oss-120b',
                    help='Local directory where model files (.safetensors, config.json, tokenizer, etc.) should live')
parser.add_argument('--max_seq_len', type=int, default=1024 * 8)
parser.add_argument('--buffer_size', type=int, default=32)
parser.add_argument('--listen_port', type=int, default=8000)
parser.add_argument("--serve", nargs="?", const='core', default=None)
parser.add_argument('--prompt', type=str, default='')
parser.add_argument('--disable_thinking', default=False, action='store_true')
parser.add_argument('--disable_fp4', default=False, action='store_true')
args = parser.parse_args()
args.hybrid_cpu = False

try:
    from safetensors.torch import safe_open, save_file
    from transformers import AutoTokenizer
except Exception as ex:
    raise Exception(f'Failed to import submodules({ex}), please install the client with:\n\n  >> {sys.executable} -m pip install "huggingface_hub[cli]" "transformers" "safetensors"')
    exit(0)

torch.backends.cuda.matmul.allow_tf32 = True

# ---------- New: local model bootstrap & HF CLI handling with verbose logging ----------

def _ts():
  return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + 'Z'

def _log(line: str, prefix: str = ""):
  msg = f"{_ts()} - {prefix}{line}" if prefix else f"{_ts()} - {line}"
  print(msg, file=sys.stderr, flush=True)  # Added flush=True for immediate output

def _has_safetensors(dir_path: str) -> bool:
  try:
    return os.path.isdir(dir_path) and any(f.endswith('.safetensors') for f in os.listdir(dir_path))
  except FileNotFoundError:
    return False

def _safe_remove_dir_contents(dir_path: str):
  """Remove everything under dir_path safely (leave the directory itself)."""
  abs_path = os.path.abspath(dir_path)
  if abs_path in ('/', ''):
    raise RuntimeError(f"Refusing to remove contents of unsafe path: '{abs_path}'")
  if not os.path.isdir(abs_path):
    return
  for entry in os.listdir(abs_path):
    full = os.path.join(abs_path, entry)
    try:
      if os.path.isfile(full) or os.path.islink(full):
        os.unlink(full)
      else:
        shutil.rmtree(full)
    except Exception as e:
      _log(f"[WARN] Failed to remove '{full}': {e}")

def _run_and_stream(cmd, env=None, prefix='[HF-CLI] '):
  """
  Run a subprocess, stream stdout+stderr lines to logs with timestamps,
  and print periodic heartbeats if the process is quiet.
  """
  _log(f"Exec: {' '.join(cmd)}", prefix=prefix)
  start = time.time()
  
  # Use PIPE for both stdout and stderr, but combine them
  proc = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      bufsize=0,  # Changed from 1 to 0 for unbuffered output
      text=True,
      universal_newlines=True,
      env=env
  )

  last_output = [time.time()]
  output_complete = threading.Event()
  
  def _reader():
    try:
      while True:
        line = proc.stdout.readline()
        if not line:
          break
        last_output[0] = time.time()
        # Normalize progress-bar carriage returns into readable updates
        line = line.rstrip('\n')
        if '\r' in line:
          # keep only the last carriage-return segment to avoid spam
          segments = line.split('\r')
          line = segments[-1]
        if line.strip():
          _log(line, prefix=prefix)
    except Exception as e:
      _log(f"[ERROR] Reader thread exception: {e}", prefix=prefix)
    finally:
      output_complete.set()
      try:
        proc.stdout.close()
      except:
        pass

  # Use non-daemon thread to ensure it completes
  t = threading.Thread(target=_reader, daemon=False)
  t.start()

  # heartbeat while running
  while True:
    rc = proc.poll()
    now = time.time()
    if now - last_output[0] > 10:  # quiet for 10s -> heartbeat
      elapsed = int(now - start)
      _log(f"(still running, {elapsed}s elapsed)", prefix=prefix)
      last_output[0] = now
    if rc is not None:
      break
    time.sleep(2)

  # Wait for reader thread to complete (with a reasonable timeout)
  # Increased timeout to ensure we capture all output
  if not output_complete.wait(timeout=30):
    _log(f"[WARN] Output reader thread did not complete within 30s", prefix=prefix)
  
  t.join(timeout=5)  # Final join to clean up thread
  
  if proc.returncode != 0:
    raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")
  elapsed = int(time.time() - start)
  _log(f"Completed in {elapsed}s", prefix=prefix)

def _ensure_latest_hf_cli(raise_on_error: bool):
  """Upgrade to latest huggingface_hub[cli] with logs."""
  try:
    cmd = [sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"]
    _run_and_stream(cmd, prefix='[HF-PIP] ')
  except Exception as e:
    _log(f"Unable to upgrade huggingface_hub[cli]: {e}", prefix='[HF-PIP] ')
    if raise_on_error:
      raise

def _find_hf_cli() -> str:
  """Return the CLI executable name ('hf' preferred), or '' if not found."""
  for candidate in ("hf", "huggingface-cli"):
    if shutil.which(candidate):
      return candidate
  return ""

def _download_with_hf_cli(repo_id: str, local_dir: str):
  """Download entire repo into local_dir (no nested repo folder) using CLI with live logs."""
  cli = _find_hf_cli()
  if not cli:
    # Final attempt: install then re-check
    _ensure_latest_hf_cli(raise_on_error=True)
    cli = _find_hf_cli()
    if not cli:
      raise RuntimeError("Hugging Face CLI not found even after installing huggingface_hub[cli].")

  # Make progress bars visible & readable in logs
  dl_env = os.environ.copy()
  # Ensure online mode and visible progress
  dl_env['HF_HUB_OFFLINE'] = '0'
  dl_env.pop('HF_HUB_DISABLE_PROGRESS_BARS', None)
  # Ensure Python output is unbuffered for subprocess
  dl_env['PYTHONUNBUFFERED'] = '1'

  # The CLI will create a small .cache dir under local_dir; that's intended.
  cmd = [cli, "download", repo_id, "--repo-type", "model", "--local-dir", local_dir]
  _log(f"Starting model download: repo='{repo_id}', dest='{local_dir}'", prefix='[HF-Download] ')

  # Disk space info (useful for visibility)
  try:
    total, used, free = shutil.disk_usage(os.path.abspath(local_dir))
    _log(f"Disk before download - total={total//(1024**3)}GiB, used={used//(1024**3)}GiB, free={free//(1024**3)}GiB",
         prefix='[HF-Download] ')
  except Exception:
    pass

  _run_and_stream(cmd, env=dl_env, prefix='[HF-Download] ')

  # Disk space post
  try:
    total, used, free = shutil.disk_usage(os.path.abspath(local_dir))
    _log(f"Disk after download  - total={total//(1024**3)}GiB, used={used//(1024**3)}GiB, free={free//(1024**3)}GiB",
         prefix='[HF-Download] ')
  except Exception:
    pass

def _summarize_safetensors(dir_path: str):
  total_bytes = 0
  count = 0
  try:
    for f in os.listdir(dir_path):
      if f.endswith('.safetensors'):
        count += 1
        try:
          total_bytes += os.path.getsize(os.path.join(dir_path, f))
        except Exception:
          pass
  except Exception:
    pass
  if count:
    mb = total_bytes / (1024 * 1024)
    _log(f"Found {count} .safetensors files ({mb:.1f} MiB) in '{dir_path}'", prefix='[HF-Verify] ')

def _ensure_local_model(path_to_model: str, hf_model: str) -> str:
  """Ensure model shards live directly in path_to_model; download if needed; log progress clearly."""
  target_dir = os.path.abspath(path_to_model)

  # Step 1: if usable local model exists, short-circuit (check before any distributed init).
  if _has_safetensors(target_dir):
    _log(f"Using existing model shards in '{target_dir}'.", prefix='[Model] ')
    _summarize_safetensors(target_dir)
    return target_dir

  # Only proceed with download if we're not in a distributed environment yet
  # or if we're rank 0
  world_rank = int(os.environ.get('RANK', 0))
  world_size = int(os.environ.get('WORLD_SIZE', 1))
  
  if world_rank == 0:
    # Step 0: ensure latest CLI
    _log(f"Ensuring latest huggingface_hub[cli]...", prefix='[HF-PIP] ')
    _ensure_latest_hf_cli(raise_on_error=False)
    
    # Step 2: prepare directory (create or wipe).
    if os.path.isdir(target_dir):
      _log(f"No .safetensors found in '{target_dir}'. Cleaning it before download.", prefix='[Model] ')
      _safe_remove_dir_contents(target_dir)
    else:
      _log(f"Creating model directory '{target_dir}'", prefix='[Model] ')
    os.makedirs(target_dir, exist_ok=True)

    # Step 3: download to local dir (no global cache pollution).
    _download_with_hf_cli(hf_model, target_dir)

    # Step 4: verify presence of safetensors in the target root.
    if not _has_safetensors(target_dir):
      raise RuntimeError(
        f"Download finished but no '.safetensors' files were found directly under '{target_dir}'. "
        f"Ensure the repo '{hf_model}' contains safetensors at its top level."
      )
    _summarize_safetensors(target_dir)

  # If in distributed mode, wait for rank 0 to complete
  if world_size > 1 and world_rank != 0:
    _log(f"Rank {world_rank} waiting for model download to complete...", prefix='[Model] ')
    # Simple file-based synchronization - wait for safetensors to appear
    max_wait = 7200  # 2 hours max wait
    start_wait = time.time()
    while not _has_safetensors(target_dir):
      if time.time() - start_wait > max_wait:
        raise RuntimeError(f"Timeout waiting for model files in '{target_dir}'")
      time.sleep(5)
    _log(f"Rank {world_rank} detected model files are ready.", prefix='[Model] ')

  # Final verification
  if not _has_safetensors(target_dir):
    raise RuntimeError(
      f"Model files are still not visible in '{target_dir}' after download. "
      f"Please check filesystem permissions and shared storage visibility."
    )
  return target_dir

# Resolve (and if needed, fetch) the model BEFORE tutel initialization
model_id = _ensure_local_model(args.path_to_model, args.hf_model)
_log(f"[INFO] Discover the model from local path: {model_id}, chosen as the default model.")

# NOW initialize tutel after model is ready
try:
  from tutel import system, net

  if 'OP_LOADER' not in os.environ:
    if not IS_HIP_EXTENSION:
      if torch.cuda.get_device_capability()[0] == 6:
        suffix = 'p100'
      elif torch.cuda.get_device_capability()[0] == 7:
        suffix = 'v100'
      elif torch.cuda.get_device_capability()[0] == 8:
        suffix = 'a100'
      elif torch.cuda.get_device_capability()[0] == 9:
        suffix = 'h100'
      elif torch.cuda.get_device_capability()[0] >= 10:
        suffix = 'b200'
      else:
        raise Exception('Device compat not recognized: %s' % torch.cuda.get_device_capability())
    else:
      if ('MI50' in torch.cuda.get_device_name()):
        suffix = 'mi50'
      else:
        suffix = 'mi300x'
    os.environ['OP_LOADER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ops.{suffix}")

  from tutel import ops
  op_loader_path = os.environ['OP_LOADER']

  if not os.path.exists(op_loader_path):
      raise Exception(f'Antares kernels are not properly configured for OP_LOADER at: {op_loader_path}')
  parallel_env = system.init_data_model_parallel(backend='gloo')
  world_rank, world_size = parallel_env.global_rank, parallel_env.global_size
  device = torch.device('cuda', parallel_env.local_rank)
  torch.cuda.set_device(device)
  peer_barrier = lambda: net.simple_all_reduce(torch.ones([1], dtype=torch.int32))

  if os.environ.get('LD_PRELOAD', '').strip():
    from tutel.impls.communicate import init_extern_nccl
    init_extern_nccl()

  def fetch_url(path, url):
    peer_barrier()
    if world_rank == 0:
      system.from_url(url, path=path)
    peer_barrier()
    return torch.from_numpy(np.load(path))
except:
  assert int(os.environ.get('WORLD_SIZE', 1)) == 1, "Failed to initialize distributed session"
  import autort
  world_rank, world_size = 0, 1
  device = autort.device()
  peer_barrier = lambda: None

  def fetch_url(path, url):
    return autort.from_npy(autort.download(path, url))

def master_print(*args, **kwargs):
  if world_rank == 0:
     print(*args, **kwargs, file=sys.stderr)

# -----------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(f'{model_id}', trust_remote_code=True)
config = json.loads(pathlib.Path(f'{model_id}/config.json').read_text())

model_type = config['model_type']
eos_token_id = tokenizer.eos_token_id if model_type != 'kimi_k2' else 163586
data_type = torch.bfloat16

if model_type in ('deepseek_v3', 'kimi_k2',):
  n_layers = int(os.environ.get('LAYER', config['num_hidden_layers']))
  n_experts = int(os.environ.get('NE', config["n_routed_experts"]))
  n_heads = config['num_attention_heads']
  log_rope_theta = math.log(config['rope_theta'])
  qk_nope_head_dim = config["qk_nope_head_dim"]
  qk_rope_head_dim = config["qk_rope_head_dim"]
  q_lora_rank = config["q_lora_rank"]
  kv_lora_rank = config["kv_lora_rank"]
  v_head_dim = config["v_head_dim"]
  n_top_k = config['num_experts_per_tok']
  check_utf8 = False
elif model_type in ('qwen3_moe', 'qwen3',):
  n_layers = int(os.environ.get('LAYER', config['num_hidden_layers']))
  n_experts = int(os.environ.get('NE', config.get("num_experts", 0)))
  n_heads = config['num_attention_heads']
  log_rope_theta = math.log(config['rope_theta'])
  n_top_k = config.get('num_experts_per_tok', 0)
  head_dim = config['head_dim']
  num_key_value_heads = config['num_key_value_heads']
  check_utf8 = True

  if config.get('quantization_config', {}).get('quant_algo', None) == 'NVFP4':
    eos_token_id = 151645
elif model_type in ('gpt_oss',):
  n_layers = int(os.environ.get('LAYER', config['num_hidden_layers']))
  n_experts = int(os.environ.get('NE', config["num_local_experts"]))
  n_heads = config['num_attention_heads']
  head_dim = config['head_dim']
  num_key_value_heads = config['num_key_value_heads']
  check_utf8 = True
else:
  raise Exception(f'Unrecognized model type: {model_type}')

eos_token_id = int(os.environ.get('EOS_TOKEN_ID', eos_token_id))

def load_to(filename, params):
  with safe_open(f'{filename}', framework='pt') as f:
    for k in f.keys():
      params[k] = f.get_tensor(k)
  return params

param = {}
state_dict = {}

for f in os.listdir(model_id):
  if f.endswith('.safetensors'):
    load_to(f'{model_id}/{f}', state_dict)

def pad_at_dim(x, dim, new_size):
  padded_shape = list(x.shape)
  if padded_shape[dim] == new_size:
    return x
  padded_shape[dim] = new_size
  y = torch.empty(padded_shape, dtype=x.dtype, device=x.device)
  y.narrow(dim, 0, x.size(dim)).copy_(x)
  y.narrow(dim, x.size(dim), new_size - x.size(dim)).zero_()
  return y


def flood(w, device=None):
  if w is None:
    return w
  device = device or w.device
  if w.dtype == torch.float8_e4m3fn:
    ws, w = getattr(w, 'scale_inv', None), w.view(torch.uint8).to(device)
    w[w == 128] = 0
    if ws is not None:
      w.scale_inv = ws.to(device)
  return w.to(device)

def load_tensor_fp8(key, device='cuda', fp8_to_bf16=False):
  w = state_dict[key].to(device)
  try:
    w.scale_inv = state_dict[key + '_scale_inv'].float().to(device)
  except:
    if w.dtype != torch.bfloat16:
      w = ops.from_float4_groupwise(w.to(device), state_dict[f'{key}_scale'].to(device), state_dict[f'{key}_scale_2'].to(device))
  if fp8_to_bf16 and w.dtype == torch.float8_e4m3fn:
    assert w.is_cuda
    w = flood(w)
    w = ops.to_bfloat16(w, w.scale_inv)
  return w

def world_slice(t, dim=0):
  if t is None:
    return t
  assert t.size(dim) % world_size == 0, f'Failed during slicing tensor of shape {list(t.shape)} to {world_size} pieces at dim-{dim}.'
  group_size = t.size(dim) // world_size
  out = t.narrow(dim, world_rank * group_size, group_size).contiguous()
  if hasattr(t, 'scale_inv'):
    assert t.scale_inv.size(dim) % world_size == 0
    group_size = t.scale_inv.size(dim) // world_size
    out.scale_inv = t.scale_inv.narrow(dim, world_rank * group_size, group_size).contiguous()
  return out

master_print(f'Loading shared weights - 0.0% ..')

if model_type in ('deepseek_v3', 'kimi_k2'):
 for k in state_dict:
  if k.startswith('model.layers.'):
    if int(k.split('.')[2]) >= n_layers:
      continue
  if 'lm_head.weight' in k:
    param[k] = state_dict[k].to(torch.bfloat16).to(device)
    param[k].scale_inv = param[k]
    if param[k].dtype == torch.bfloat16:
      param[k], fp8scal = ops.to_float8_block(param[k])
      param[k].scale_inv = fp8scal
    continue
  if 'model.embed_tokens.weight' in k or 'norm' in k:
    param[k] = state_dict[k].to(torch.bfloat16).to(device)
    continue
  if '.rotary_emb.inv_freq' in k:
    continue
  if 'down_proj.' in k or 'gate_proj.' in k or 'up_proj.' in k:
    continue
  if '_exps' in k or k.endswith('_scale_inv') or 'kv_a_proj_with_mqa' in k:
    continue
  if 'self_attn.kv_b_proj.' in k or 'self_attn.q_b_proj.' in k:
    param[k] = world_slice(load_tensor_fp8(k, fp8_to_bf16=True), dim=0)
    continue
  if 'self_attn.o_proj.' in k:
    param[k] = flood(world_slice(load_tensor_fp8(k, 'cpu'), dim=1), device)
    if param[k].dtype == torch.bfloat16:
      param[k], fp8scal = ops.to_float8_block(param[k])
      param[k].scale_inv = fp8scal
    continue
  if 'self_attn.q_a_proj.weight' in k:
    Q = load_tensor_fp8(k, 'cpu')
    KV = load_tensor_fp8(k.replace('q_a_proj', 'kv_a_proj_with_mqa'), 'cpu')
    qkv = k.replace('q_a_proj', 'qkv_a_proj')
    param[qkv] = flood(torch.cat([Q.view(torch.uint8), KV.view(torch.uint8)], dim=0).view(Q.dtype)).to(device)
    if hasattr(Q, 'scale_inv'):
      param[qkv].scale_inv = torch.cat([Q.scale_inv, KV.scale_inv], dim=0).to(device)
    if param[qkv].dtype == torch.bfloat16:
      param_dim_size = param[qkv].size(0)
      padded_w = torch.nn.functional.pad(param[qkv], (0, 0, 0, (param_dim_size + 127) // 128 * 128 - param_dim_size))
      param[qkv], fp8scal = ops.to_float8_block(padded_w)
      param[qkv] = param[qkv].narrow(0, 0, param_dim_size)
      param[qkv].scale_inv = fp8scal
    continue
  if '.mlp.gate.e_score' in k or '.mlp.gate.weight' in k:
    param[k] = state_dict[k][:n_experts].contiguous().to(torch.bfloat16).to(device)
    continue
  master_print('>>>', k, state_dict[k].shape, state_dict[k].dtype, state_dict[k].view(-1)[:5]); exit(0)

elif model_type in ('qwen3_moe', 'qwen3'):

 for k in state_dict:
  if k.startswith('model.layers.'):
    if int(k.split('.')[2]) >= n_layers:
      continue
  if 'norm' in k:
    param[k] = state_dict[k].to(torch.bfloat16).to(device)
    continue
  if 'down_proj.' in k or 'gate_proj.' in k or 'up_proj.' in k or k.endswith('_scale_inv') or k.endswith('_scale') or k.endswith('_scale_2'):
    continue
  if '.mlp.gate.weight' in k:
    param[k] = state_dict[k][:n_experts].contiguous().to(torch.bfloat16).to(device)
    continue
  if 'self_attn.o_proj.' in k:
    param[k] = flood(world_slice(load_tensor_fp8(k, 'cpu'), dim=1), device)
    continue
  if 'lm_head.weight' in k or 'model.embed_tokens.weight' in k:
    param[k] = state_dict[k].contiguous().to(torch.bfloat16).to(device)
    continue
  if 'attn.k_' in k or 'attn.v_' in k:
    continue
  if 'attn.q_' in k:
    q_param = world_slice(load_tensor_fp8(k, fp8_to_bf16=True))
    k_param = world_slice(load_tensor_fp8(k.replace('.q_', '.k_'), fp8_to_bf16=True).view(num_key_value_heads, 1, head_dim, -1).repeat(1, max(1, world_size // num_key_value_heads), 1, 1).flatten(0, 2))
    v_param = world_slice(load_tensor_fp8(k.replace('.q_', '.v_'), fp8_to_bf16=True).view(num_key_value_heads, 1, head_dim, -1).repeat(1, max(1, world_size // num_key_value_heads), 1, 1).flatten(0, 2))
    param[k.replace('.q_', '.qkv_')] = torch.cat([q_param, k_param, v_param], dim=0)
    continue
  master_print('>>>', k, state_dict[k].shape, state_dict[k].dtype, state_dict[k].view(-1)[:5]); exit(0)

elif model_type in ('gpt_oss'):
 for k in state_dict:
  if k.startswith('model.layers.'):
    if int(k.split('.')[2]) >= n_layers:
      continue
  if 'model.embed_tokens.weight' in k or '.router.' in k or 'norm' in k:
    param[k] = state_dict[k].contiguous().to(torch.bfloat16).to(device)
    continue
  if 'lm_head.weight' in k:
    param[k] = state_dict[k].contiguous().to(torch.bfloat16).to(device)
    ''' original_size = param[k].size(1)
    param[k] = pad_at_dim(param[k], 1, (param[k].size(1) + 127) // 128 * 128)
    param[k], scale_inv = ops.to_float8_block(param[k])
    param[k] = param[k].narrow(1, 0, original_size).contiguous()
    param[k].scale_inv = scale_inv '''
    continue
  if '.sinks' in k:
    param[k] = world_slice(state_dict[k].contiguous().to(torch.bfloat16).to(device))
    continue
  if 'attn.k_' in k or 'attn.v_' in k or '.bias' in k or '.experts.down_' in k or '.experts.gate_up_' in k:
    continue
  if 'attn.q_' in k:
    q_param = world_slice(load_tensor_fp8(k, fp8_to_bf16=True))
    k_param = world_slice(load_tensor_fp8(k.replace('.q_', '.k_'), fp8_to_bf16=True).view(num_key_value_heads, 1, head_dim, -1).repeat(1, max(1, world_size // num_key_value_heads), 1, 1).flatten(0, 2))
    v_param = world_slice(load_tensor_fp8(k.replace('.q_', '.v_'), fp8_to_bf16=True).view(num_key_value_heads, 1, head_dim, -1).repeat(1, max(1, world_size // num_key_value_heads), 1, 1).flatten(0, 2))
    q_param.bias = world_slice(load_tensor_fp8(k.replace('.weight', '.bias'), fp8_to_bf16=True))
    k_param.bias = world_slice(load_tensor_fp8(k.replace('.q_', '.k_').replace('.weight', '.bias'), fp8_to_bf16=True))
    v_param.bias = world_slice(load_tensor_fp8(k.replace('.q_', '.v_').replace('.weight', '.bias'), fp8_to_bf16=True))
    qkv_param = torch.cat([q_param, k_param, v_param], dim=0)
    qkv_param.bias = torch.cat([q_param.bias, k_param.bias, v_param.bias], dim=0)
    param[k.replace('.q_', '.qkv_')] = qkv_param
    continue
  if 'self_attn.o_proj.' in k:
    param[k] = flood(world_slice(load_tensor_fp8(k, 'cpu'), dim=1), device)
    param[k].bias = flood(load_tensor_fp8(k.replace('.weight', '.bias'), 'cpu'), device) / world_size
    continue
  master_print('>>>', k, state_dict[k].shape, state_dict[k].dtype, state_dict[k].view(-1)[:5])

else:
  raise Exception(f'Unrecognized model type: {model_type}')

token_emb = param['model.embed_tokens.weight']
gate_moe = [param.get(f'model.layers.{i}.mlp.gate.weight', param.get(f'model.layers.{i}.mlp.router.weight', None)) for i in range(n_layers)]


def load_expert_weight(prefs, dim=None, dev='cpu'):
  if type(prefs) not in (tuple, list):
    prefs = [prefs]
  ws, ss, mi, mw = [], [], [], []
  use_fp4 = (not args.disable_fp4) and (not args.hybrid_cpu)

  for pref in prefs:
    if f'{pref}.weight_scale_inv' in state_dict:
      w, s = state_dict[f'{pref}.weight'], state_dict[f'{pref}.weight_scale_inv']
      if dim is not None:
        assert w.dtype == torch.float8_e4m3fn
        if s.size(dim) % world_size != 0:
          block_size = w.size(dim) // s.size(dim)
          s = pad_at_dim(s, dim, (s.size(dim) + world_size - 1) // world_size * world_size)
          w = pad_at_dim(w, dim, s.size(dim) * block_size)
        w, s = flood(world_slice(w.view(torch.uint8), dim=dim).view(w.dtype), device=device), world_slice(s, dim=dim).float().to(device)
      if use_fp4:
        w, s, o = ops.to_float4_groupwise(ops.to_bfloat16(w, s))
        ws += [w]
        ss += [s]
        mi += [o.view(1)]
        mw += [o.view(1)]
        torch.cuda.synchronize()
        del w, s, o, state_dict[f'{pref}.weight'], state_dict[f'{pref}.weight_scale_inv']
        continue
      ws += [w]
      ss += [s]
      del state_dict[f'{pref}.weight'], state_dict[f'{pref}.weight_scale_inv']
      continue
    w, s = state_dict[f'{pref}.weight'], state_dict.get(f'{pref}.weight_scale', None)
    if dim is not None:
      w, s = world_slice(w, dim=dim), flood(world_slice(s, dim=dim))
    if w.dtype == torch.bfloat16 and use_fp4 and len([_ for _ in gate_moe if _ is not None]) > 0:
      w, s, o = ops.to_float4_groupwise(w)
      ws += [w]
      ss += [s]
      mi += [o.view(1)]
      mw += [o.view(1)]
      torch.cuda.synchronize()
      del w, s, o, state_dict[f'{pref}.weight']
      continue
    if args.hybrid_cpu and str(dev).startswith('cpu'):
      assert world_size == 1, "World size must be 1 under hybrid-CPU mode."
      w = ops.from_float4_groupwise(w, s, state_dict[f'{pref}.weight_scale_2'])
      s = None
    ws += [w]
    if s is None:
      continue
    ss += [s]
    mi += [state_dict[f'{pref}.input_scale'].view(1)]
    mw += [state_dict[f'{pref}.weight_scale_2'].view(1)]
  if not ss:
    ws = torch.cat(ws, dim=0).unsqueeze(0).to(dev)
  elif not mw:
    ws = torch.cat(ws, dim=0).unsqueeze(0).to(dev).view(torch.float8_e4m3fn)
    ws.scale_inv = torch.cat(ss, dim=0).unsqueeze(0).to(dev)
  else:
    ws = torch.cat(ws, dim=0).unsqueeze(0).to(dev)
    ws.scale_inv = torch.cat(ss, dim=0).unsqueeze(0).to(dev).view(torch.float8_e4m3fn)
    ws.meta_input = torch.cat(mi, dim=0).unsqueeze(0).to(dev)
    ws.meta_weight = torch.cat(mw, dim=0).unsqueeze(0).to(dev)
  return ws


def load_experts():
  gate_up_p, down_p = [], []

  for i in range(n_layers):
    if gate_moe[i] is None:
      gate_up_proj = load_expert_weight([f'model.layers.{i}.mlp.gate_proj', f'model.layers.{i}.mlp.up_proj'], dim=-2, dev=device)
      down_proj = load_expert_weight(f'model.layers.{i}.mlp.down_proj', dim=-1, dev=device)
      gate_up_p += [gate_up_proj]
      down_p += [down_proj]
      continue
    master_print(f'Loading expert weights - {(i + 1) / n_layers * 100:.1f}% ..')

    if args.hybrid_cpu:
      continue

    def pack(proj, device):
      if str(device).startswith('cpu'):
        copy_to_device = torch.cat
      else:
        copy_to_device = ops.copy_to_device

      if proj[0].dtype == torch.bfloat16:
        local = copy_to_device(proj)
      elif not hasattr(proj[0], 'meta_weight'):
        local = copy_to_device(proj).view(torch.float8_e4m3fn)
        local.scale_inv = copy_to_device([_.scale_inv for _ in proj])
      else:
        local = copy_to_device(proj)
        local.scale_inv = copy_to_device([_.scale_inv for _ in proj]).view(torch.float8_e4m3fn)
        local.meta_input = torch.cat([_.meta_input.view(1, -1) for _ in proj], dim=0).to(device)
        local.meta_weight = torch.cat([_.meta_weight.view(1, -1) for _ in proj], dim=0).to(device)
      return local

    local_device = device if not args.hybrid_cpu else 'cpu'
    if f'model.layers.{i}.mlp.experts.gate_up_proj_blocks' in state_dict:
      def load_mxfp4(prefix, fn):
        p = fn(state_dict[f'{prefix}_blocks']).to(local_device)
        p.scales, p.bias = fn(state_dict[f'{prefix}_scales']).to(local_device), fn(state_dict[f'{prefix}_bias']).to(local_device)
        return p

      group_size = (90 + world_size - 1) // world_size * world_size
      gate_up_p += [load_mxfp4(f'model.layers.{i}.mlp.experts.gate_up_proj', lambda x: world_slice(pad_at_dim(x, 1, group_size << 6), dim=1))]
      down_p += [load_mxfp4(f'model.layers.{i}.mlp.experts.down_proj', lambda x: world_slice(pad_at_dim(x, 2, group_size), dim=2) if x.dim() >= 3 else x / world_size)]
    else:
      gate_up_proj = [load_expert_weight([f'model.layers.{i}.mlp.experts.{ID}.gate_proj', f'model.layers.{i}.mlp.experts.{ID}.up_proj'], dim=-2) for ID in range(n_experts)]
      down_proj = [load_expert_weight(f'model.layers.{i}.mlp.experts.{ID}.down_proj', dim=-1) for ID in range(n_experts)]
      try:
        gate_up_proj += [load_expert_weight([f'model.layers.{i}.mlp.shared_experts.gate_proj', f'model.layers.{i}.mlp.shared_experts.up_proj'], dim=-2)]
        down_proj += [load_expert_weight(f'model.layers.{i}.mlp.shared_experts.down_proj', dim=-1)]
      except:
        pass

      gate_up_p += [pack(gate_up_proj, device=local_device)]
      down_p += [pack(down_proj, device=local_device)]
      del gate_up_proj, down_proj

  return gate_up_p, down_p

gate_up_p, down_p = load_experts()

del state_dict

batch = int(os.environ.get('BATCH', 1))
args.max_seq_len = (args.max_seq_len + args.buffer_size - 1) // args.buffer_size * args.buffer_size
assert args.max_seq_len % args.buffer_size == 0
buffer_data = torch.empty([args.buffer_size], dtype=torch.int64, device=device)
GLOBAL_INIT_FN = set()

token_emb = token_emb.view(token_emb.size(0), -1)

master_print('Synchronizing with other peers..')
peer_barrier()

try:
  sigp = torch.ops.tutel_ops.uncached_empty([8192 * 16], torch.int32)
  sigp = torch.ops.tutel_ops.uncached_exchange(sigp[0], net.simple_all_gather(sigp[1]), world_rank)
  buffer = torch.ops.tutel_ops.uncached_empty([batch, token_emb.size(-1)], torch.bfloat16)
  buffer = torch.ops.tutel_ops.uncached_exchange(buffer[0], net.simple_all_gather(buffer[1]), world_rank)
except Exception as e:
  if world_rank == 0:
     master_print(f'\n{e}')
  exit(1)

if model_type in ('deepseek_v3', 'kimi_k2'):
  module = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./models/deepseek_moe.py")
elif model_type in ('qwen3_moe', 'qwen3'):
  module = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./models/qwen3_moe.py")
elif model_type in ('gpt_oss'):
  module = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./models/gpt_oss_moe.py")
else:
  raise Exception(f'Unrecognized model type: {model_type}')

exec(compile(open(module).read(), filename=module, mode='exec'))


logits = torch.zeros([batch, 1, token_emb.size(0)], dtype=torch.bfloat16, device=device)
token_in = torch.ones([batch, 1], dtype=torch.int64, device=device)
token_range = torch.tensor([0, 1], dtype=torch.int32, device=device)

replay = lambda position_id: forward_fn(token_in, token_range, logits, position_id)

if use_cugraph:
  gs = []
  mapping_table = np.zeros([args.max_seq_len + 2], dtype='int32')
  mapping_graph = [256, 512, 1024, 2048, 4096] + ([args.max_seq_len] if args.max_seq_len > 4096 else [])

  for k, v in enumerate(mapping_graph):
    for i in range(3):
      replay(v - 1)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
      replay(v - 1)
    torch.cuda.synchronize()
    gs.append(g)
    mapping_table[v + 1] = 1

  mapping_table = list(mapping_table.cumsum())
  replay = lambda pos: gs[mapping_table[pos]].replay()

def forward(token, pos):
  token_range[1] = pos + 1
  token_in.copy_(token)
  replay(pos)
  return logits

def benchmark(use_ones=False, n_steps=100):
  prof_type = int(os.environ.get('PROF', 0))
  if prof_type:
    position_id = prof_type - 1
    import autort
    # autort.perform(lambda: forward(token, position_id), n_steps)
    autort.perform(lambda: replay(position_id) or logits, n_steps)

  for seq_len in [64, 256, 1024, 4096] + ([args.max_seq_len] if args.max_seq_len > 4096 else []):
    if seq_len <= args.max_seq_len:
      with torch.no_grad():
        position_id = seq_len - 1
        token = torch.randint(1, 2 if use_ones else 4096, [batch, 1], dtype=torch.int64, device=device)
        forward(token, position_id)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(n_steps):
          replay(position_id)
        torch.cuda.synchronize()
        t_stop = time.perf_counter()
        master_print('>> Decode TPS Benchmark (bsz = %d, output_ctx = %d, n_gpus = %d): %.4f tokens/sec' % (batch, seq_len, world_size, token.numel() * n_steps / (t_stop - t_start),))

  if prof_type:
    exit(0)


def generate_partial(prompt_tokens, max_tokens=None, temperature=0, screen_display=True):
  max_tokens = max_tokens or args.max_seq_len
  temperature /= 1000.0

  progress_mask = ['\r-', '\r\\', '\r|', '\r/']

  with torch.no_grad():
    pos, token = 0, prompt_tokens[0].view(-1, 1).contiguous()
    batch, seq = token.size(0), token.size(1)
    visible_point = len(prompt_tokens) - 1
    for fn in GLOBAL_INIT_FN:
      fn()
    peer_barrier()
    torch.cuda.synchronize()
    master_print('*******************************************************')
    try:
      master_print(tokenizer.decode(prompt_tokens).strip())
    except:
      master_print('Prompting ..')
    master_print('*******************************************************')

    t_start = time.perf_counter()
    last_token = []

    def token_flush():
      decoded_word = tokenizer.decode(last_token)
      last_token.clear()

      if screen_display:
        sys.stderr.write(decoded_word)
        sys.stderr.flush()
      return decoded_word

    while pos < max_tokens:
      for _ in range(args.buffer_size):
        logits = forward(token, pos + _)
        if pos + _ + 1 < len(prompt_tokens):
          next_token = prompt_tokens[pos + _ + 1].view(batch, seq)
        else:
          if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1).view(batch, seq)
          else:
            next_token = torch.multinomial(torch.softmax(logits.view(batch * seq, -1) / temperature, dim=1), num_samples=batch * seq).view(batch, seq)
        token = next_token.contiguous()
        buffer_data[_] = token[-1]

      visible_offset = max(0, visible_point - pos)
      pos += args.buffer_size

      if visible_offset >= args.buffer_size:
        continue

      spill_tokens = buffer_data[visible_offset:].cpu().tolist()
      try:
        _ = spill_tokens.index(eos_token_id)
      except ValueError:
        _ = len(spill_tokens)

      if world_rank == 0 and _ > 0:
        if check_utf8:
          last_token += spill_tokens[:_]
          if (tokenizer.decode(spill_tokens[_ - 1]).encode('utf-8')[-1]) <= 127:
            yield token_flush()
        else:
          last_token = spill_tokens[:_]
          yield token_flush()
      if _ < len(spill_tokens):
        if last_token:
          yield token_flush()
        break
    t_stop = time.perf_counter()

  yield (t_stop - t_start), pos + 1

def generate(*args, **kwargs):
  buffers = []
  for _ in generate_partial(*args, **kwargs):
    if isinstance(_, str):
      buffers.append(_)
  return ''.join(buffers), *_

def token_encode(user_prompt, system_prompt=None):
  text = tokenizer.apply_chat_template(
      [{"role": "user", "content": user_prompt}] + ([] if system_prompt is None else [{"role": "system", "content": system_prompt}]),
      tokenize=False,
      add_generation_prompt=True,
      reasoning_effort='low',
      enable_thinking=not args.disable_thinking
  )
  return tokenizer([text], return_tensors="pt")['input_ids'].view(-1)

prompt_cnt = torch.empty([3], dtype=torch.int64)
tokens_buf = torch.empty([args.max_seq_len], dtype=torch.int64)

from fastapi import FastAPI, Request, Response

def router_fn(app):
  from fastapi.responses import StreamingResponse
  from starlette.background import BackgroundTask

  if args.serve == 'webui':
    from open_webui.main import log as log_message
    log_message = log_message.info
  else:
    log_message = master_print

  import asyncio
  app.global_lock = asyncio.Lock()
  app.model_name = "(Model) " + os.path.basename(model_id)
  app.task_pool = iter(set())

  @app.get("/v1/models")
  @app.get("/api/tags")
  @app.get("/api/ps")
  async def get_serving_models():
    return {"models":[{"name":app.model_name, "model":model_id + ":latest", "details":{}}], "data": [{"id": app.model_name, "object": "model"}], "object": "list"}

  @app.post("/chat")
  @app.post("/v1/chat/completions")
  @app.post("/api/chat")
  async def stream_chat(request: Request):
    prompt_messages = await request.json()
    if len(prompt_messages) == 1 and 'text' in prompt_messages:
      prompt_messages = {"messages": [{"role": "user", "content": prompt_messages["text"]}]}

    max_tokens = min(int(prompt_messages.get('max_tokens', args.max_seq_len)), args.max_seq_len)
    scope_route = request.scope.get("route").path
    enable_stream = (scope_route != "/v1/chat/completions")

    async def background_cleanup():
      try:
        next(app.task_pool)
        print('>> Processing Background Cleanup..')
        while True:
          next(app.task_pool)
      except:
        pass
      app.task_pool = iter(set())

    async def generate_data():
      async with app.global_lock:
        prompt_system = None
        for message in prompt_messages['messages'][-2:]:
          if message.get('role', 'user') == 'system':
            prompt_system = message['content']
          else:
            prompt_user = message['content']

        if prompt_user.endswith('</chat_history>') and not prompt_messages['stream']:
          yield json.dumps({})
          return

        tokens = token_encode(prompt_user.strip(), system_prompt=prompt_system)
        temperature = int(1000 * abs(float(prompt_messages.get('temperature', 0))))

        await background_cleanup()
        prompt_cnt[0] = tokens.numel()
        prompt_cnt[1] = max_tokens
        prompt_cnt[2] = temperature
        net.simple_broadcast(prompt_cnt)
        net.simple_broadcast(tokens)

        app.task_pool = generate_partial(tokens.to(device), max_tokens=max_tokens, temperature=temperature, screen_display=not enable_stream)
        response_buffers = []
        for response in app.task_pool:
          if isinstance(response, str):
            if enable_stream:
              if scope_route == "/chat":
                yield response
              else:
                json_out = {"model": model_id + ":latest","message":{"role":"assistant","content": response},"done": False}
                yield json.dumps(json_out) + '\n'
              await asyncio.sleep(0)
            else:
              response_buffers.append(response)

        message = { "role": "assistant", "content": ''.join(response_buffers)}
        time_cost, total_tokens = response
        usage = {"total_tokens": total_tokens, "time_cost": time_cost}

        log_message(f'\x1b[33;20m<<<< Complete {total_tokens} tokens in {time_cost:.3f} sec (Decode TPS = {total_tokens / time_cost:.3f}) >>>>\x1b[0m')
        if scope_route == "/api/chat":
          json_out = {"model":model_id + ":latest","message": message,"done": True, "usage": usage}
          yield json.dumps(json_out)
        elif scope_route == "/v1/chat/completions":
          json_out = {"id": model_id + ":latest", "choices": [{"index": 0, "finish_reason": "stop", "message": message}], "usage": usage}
          yield json.dumps(json_out)
      yield '\n'

    return StreamingResponse(generate_data(), media_type="application/json") # background=BackgroundTask(background_cleanup)

FastAPI.router_fn = router_fn

def serve():
  if world_rank == 0:
    addr = '0.0.0.0'
    master_print(f'''
Start listening on {addr}:{args.listen_port}. Request examples:

  >> curl -X POST http://{addr}:{args.listen_port}/chat -d '{{"text": "Given x + 1/x = 1, calculate x^2025 + 1/(x^2025)."}}'
  >> curl -X POST http://{addr}:{args.listen_port}/api/chat -d '{{"messages": [{{"content": "Given x + 1/x = 1, calculate x^2025 + 1/(x^2025)."}}]}}'
  >> curl -X POST http://{addr}:{args.listen_port}/v1/chat/completions -d '{{"messages": [{{"content": "Given x + 1/x = 1, calculate x^2025 + 1/(x^2025)."}}], "max_tokens": 1024 }}'
''')
    if args.serve == 'webui':
      os.environ['WEBUI_AUTH'] = '0'
      from open_webui import serve
      serve(host=addr, port=args.listen_port)
    else:
      import uvicorn
      app = FastAPI()
      FastAPI.router_fn(app)
      uvicorn.run(app, host=addr, port=args.listen_port)
  else:
    with torch.no_grad():
      while True:
        net.simple_broadcast(prompt_cnt)
        prompt_size, max_tokens, temperature = prompt_cnt.cpu().tolist()
        tokens = tokens_buf.narrow(0, 0, prompt_size)
        net.simple_broadcast(tokens)
        generate(tokens.to(device), max_tokens=max_tokens, temperature=temperature)

if __name__ == '__main__':
  user_prompt = args.prompt.strip()
  master_print()
  if user_prompt:
    generate(token_encode(user_prompt).to(device))
    master_print()
  benchmark()
  if args.serve is None and not user_prompt:
     args.serve = "core"
  if args.serve is not None:
    serve()
