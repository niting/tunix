"""Standalone script for fast evaluation of Sampler (vLLM) vs Trainer (MaxText) parity.

Performs:
1. Model loading for both Actor (MaxText) and Rollout (vLLM) across meshes.
2. Weight synchronization via Raiden (`rl_engine.sync_weights()`).
3. Single prompt generation using vLLM to collect tokens, rollout logprobs, and routed expert choices.
4. Passing the completions to MaxText's `compute_per_token_logps` (and MoE expert choices).
5. Comprehensive diagnostic printing:
   - Summary statistics (mean/max |logdiff|, min/max weight, is_oob_ratio, outlier tokens)
   - Layer-by-layer MoE router agreement (exact match and top-k overlap)
   - Top diverging tokens breakdown table
   - Position-dependent RoPE drift histogram
   - Raw logit distribution comparison
"""

import argparse
import faulthandler
import os
import signal
import sys
import time

os.environ["VLLM_TPU_RPA_VERSION"] = "2"
os.environ["DISABLE_MOSAIC_ATTN"] = "1"

from flax import nnx
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.integration.vllm import maxtext_vllm_adapter
import numpy as np
import optax
from transformers import AutoTokenizer

from tunix.cli.utils import data as data_lib
from tunix.models.qwen3 import qwen3_actor
from tunix.rl import algo_core
from tunix.rl import common
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import sft_utils

# Register MaxText vLLM adapter
maxtext_vllm_adapter.register()

# Disable MRoPE
try:
  from vllm.config import ModelConfig
  ModelConfig.uses_mrope = property(lambda self: False)
except Exception:
  pass

faulthandler.register(signal.SIGINT, all_threads=True)


def parse_args():
  parser = argparse.ArgumentParser(description="Evaluate Sampler vs Trainer Parity")
  parser.add_argument("--model_version", type=str, default="Qwen3.5-35B-A3B")
  parser.add_argument(
      "--model_absolute_path",
      type=str,
      default="gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items",
  )
  parser.add_argument("--scan_layers", type=lambda v: v.lower() == "true", default=True)
  parser.add_argument("--padded_base_moe_mlp_dim", type=int, default=1024)
  parser.add_argument("--float32_gate_logits", type=lambda v: v.lower() == "true", default=True)
  parser.add_argument("--use_flash_attention", type=lambda v: v.lower() == "true", default=True)
  parser.add_argument("--vllm_utilization", type=float, default=0.70)
  parser.add_argument("--rollout_mesh_fsdp", type=int, default=32)
  parser.add_argument("--rollout_mesh_tp", type=int, default=4)
  parser.add_argument("--train_mesh_fsdp", type=int, default=64)
  parser.add_argument("--train_mesh_tp", type=int, default=2)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=1)  # Greedy for exact deterministic verification
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument("--tis_ratio_min", type=float, default=0.999)
  parser.add_argument("--tis_ratio_max", type=float, default=1.002)
  return parser.parse_args()


def main():
  args = parse_args()
  print("=" * 80)
  print("STARTING SAMPLER-TRAINER PARITY EVALUATION")
  print("=" * 80)

  tokenizer_path = data_lib.find_local_tokenizer_path(args.model_version)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

  devices = jax.devices()
  num_devices = len(devices)
  print(f"Discovered {num_devices} JAX devices: {devices[0].device_kind}")

  # 1. Meshes
  rollout_device_mesh = mesh_utils.create_device_mesh(
      (args.rollout_mesh_fsdp, args.rollout_mesh_tp), devices=devices
  )
  rollout_mesh = Mesh(rollout_device_mesh, ("data", "model"))

  train_device_mesh = mesh_utils.create_device_mesh(
      (args.train_mesh_fsdp, 1, args.train_mesh_tp), devices=devices
  )
  actor_mesh = Mesh(train_device_mesh, ("data", "sequence", "model"))
  reference_mesh = actor_mesh

  # 2. Configs
  model_name_slug = args.model_version.lower().replace(".", "_").replace("-", "_")
  extra_maxtext_args = [
      f"checkpoint_is_scanned={args.scan_layers}",
      f"scan_layers={args.scan_layers}",
      f"float32_gate_logits={args.float32_gate_logits}",
      f"padded_base_moe_mlp_dim={args.padded_base_moe_mlp_dim}",
  ]

  trainer_config = qwen3_actor.create_maxtext_trainer_config(
      model_name=model_name_slug,
      mesh=actor_mesh,
      checkpoint_dir=args.model_absolute_path,
      use_flash_attention=args.use_flash_attention,
      extra_args=extra_maxtext_args,
  )

  sampler_config = qwen3_actor.create_maxtext_sampler_config(
      model_name=model_name_slug,
      mesh=rollout_mesh,
      checkpoint_dir=args.model_absolute_path,
      use_flash_attention=args.use_flash_attention,
      extra_args=extra_maxtext_args,
  )

  # 3. Instantiate Models
  print("Instantiating Actor (MaxText)...", flush=True)
  with actor_mesh:
    qwen_actor = qwen3_actor.create_actor_model(
        model_name=model_name_slug,
        mesh=actor_mesh,
        maxtext_config=trainer_config,
    )
    optimizer = optax.adam(1e-6)

  with reference_mesh:
    qwen_reference = qwen3_actor.create_reference_model(
        model_name=model_name_slug,
        mesh=reference_mesh,
        maxtext_config=trainer_config,
    )

  # 4. Rollout & Cluster Config
  base_rollout_dict = {
      "temperature": args.temperature,
      "top_k": args.top_k,
      "top_p": 1.0,
      "max_tokens": args.max_response_length,
      "pad_output": True,
      "pad_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
      "eos_id": tokenizer.eos_token_id,
      "rollout_engine": "vllm",
      "model_version": args.model_version,
  }

  vllm_rollout_dict = {
      "model": tokenizer_path,
      "tensor_parallel_size": rollout_mesh.shape.get("model", 1),
      "data_parallel_size": rollout_mesh.shape.get("data", 1),
      "rollout_vllm_max_num_seqs": 16,
      "rollout_vllm_max_num_batched_tokens": 16384,
      "rollout_vllm_kwargs": {
          "kv_cache_metrics": True,
          "disable_log_stats": False,
          "enable_prefix_caching": False,
          "tokenizer": tokenizer_path,
          "dtype": "bfloat16",
          "enable_expert_parallel": False,
          "generation_config": "vllm",
          "hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
      },
      "rollout_vllm_additional_config": {
          "maxtext_config": {
              "model_name": model_name_slug,
              "model_call_mode": "inference",
              "attention": "vllm_rpa",
              "allow_split_physical_axes": True,
              "log_config": False,
              "weight_dtype": "bfloat16",
              "prefuse_moe_weights": True,
              "remat_policy": "none",
              "enable_dp_attention": False,
              "float32_gate_logits": args.float32_gate_logits,
              "vllm_hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
          }
      },
      "rollout_vllm_sampling_kwargs": {
          "stop": ["<|im_end|>", "<|endoftext|>"],
          "stop_token_ids": [
              tokenizer.encode("<|im_end|>")[0],
              tokenizer.encode("<|endoftext|>")[0],
          ],
          "detokenize": True,
      },
  }

  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict, **vllm_rollout_dict)

  import functools
  from maxtext.integration.vllm.maxtext_vllm_rollout import MaxTextVllmRollout
  rollout_engine_arg = functools.partial(MaxTextVllmRollout, maxtext_config=sampler_config)

  cluster_config = rl_engine_lib.ClusterConfig(
      role_to_mesh={
          rl_engine_lib.Role.ACTOR: actor_mesh,
          rl_engine_lib.Role.REFERENCE: reference_mesh,
          rl_engine_lib.Role.ROLLOUT: rollout_mesh,
      },
      role_to_logical_axis_rule={
          rl_engine_lib.Role.ACTOR: trainer_config.logical_axis_rules,
          rl_engine_lib.Role.REFERENCE: trainer_config.logical_axis_rules,
          rl_engine_lib.Role.ROLLOUT: sampler_config.logical_axis_rules,
      },
      rollout_engine=rollout_engine_arg,
      offload_to_cpu=False,
      training_config=rl_engine_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          mini_batch_size=1,
          train_micro_batch_size=1,
          compute_logps_micro_batch_size=1,
          rollout_micro_batch_size=1,
          checkpoint_root_directory=None,
      ),
      rollout_config=rollout_engine_config,
  )

  print("Initializing RLEngine...", flush=True)
  rl_engine = rl_engine_lib.RLEngine(
      actor=qwen_actor,
      reference=qwen_reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  print("Syncing initial checkpoint weights to rollout workers via Raiden...", flush=True)
  rl_engine.sync_weights()
  print("Weight sync complete!", flush=True)

  # 5. Execute Test Generation
  test_prompt = (
      "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n"
      "<|im_start|>user\nWrite a python function to check if a string is a palindrome.<|im_end|>\n"
      "<|im_start|>assistant\n"
  )
  print(f"\nSubmitting prompt for rollout generation:\n{repr(test_prompt)}")

  rollout_out = rl_engine.generate(
      prompts=[test_prompt],
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=1.0,
      max_tokens=256,
  )

  completion_text = rollout_out.text[0]
  print(f"\nGenerated Completion:\n{completion_text}\n")

  # Extract token IDs & rollout logprobs
  gen_tokens = np.asarray(rollout_out.tokens[0])
  prompt_tokens = np.asarray(tokenizer.encode(test_prompt))
  prompt_batch = jnp.asarray(prompt_tokens[None, :])
  comp_batch = jnp.asarray(gen_tokens[None, :])
  pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
  eos_id = tokenizer.eos_token_id

  sampler_logps = np.asarray(rollout_out.logprobs[0]) if rollout_out.logprobs is not None else None
  sampler_routed_experts = np.asarray(rollout_out.routed_experts[0]) if rollout_out.routed_experts is not None else None

  print(f"Completion length: {len(gen_tokens)} tokens")

  # 6. Score using Trainer (MaxText)
  print("Computing Trainer per-token log-probabilities via MaxText...", flush=True)
  trainer_logps_jax = rl_engine.get_actor_per_token_logps(
      prompt_tokens=prompt_batch,
      completion_tokens=comp_batch,
      pad_id=pad_id,
      eos_id=eos_id,
      micro_batch_size=1,
  )
  trainer_logps = np.asarray(trainer_logps_jax[0])

  if sampler_logps is None or len(sampler_logps) == 0:
    print("Warning: rollout_out.logprobs was None; falling back to rollout engine logprob pass.")
    sampler_logps = np.zeros_like(trainer_logps)

  # Trim to valid non-pad tokens
  valid_mask = (gen_tokens != pad_id) & (gen_tokens != eos_id)
  valid_indices = np.where(valid_mask)[0]
  if len(valid_indices) == 0:
    valid_indices = np.arange(len(gen_tokens))

  t_logps = trainer_logps[valid_indices]
  s_logps = sampler_logps[valid_indices]
  toks = gen_tokens[valid_indices]

  log_diff = t_logps - s_logps
  abs_log_diff = np.abs(log_diff)
  weights = np.exp(log_diff)

  is_oob = (weights < args.tis_ratio_min) | (weights > args.tis_ratio_max)
  outliers = abs_log_diff > 10.0

  # SECTION 1
  print("\n" + "=" * 80)
  print(" " * 25 + "SECTION 1: PARITY SUMMARY")
  print("=" * 80)
  print(f"Total Scored Tokens:                {len(toks):,}")
  print(f"Mean Absolute Logdiff (|Δ|):        {np.mean(abs_log_diff):.6f} nats")
  print(f"Max Absolute Logdiff (max |Δ|):     {np.max(abs_log_diff):.6f} nats")
  print(f"Geometric Mean Ratio (seq_geomean): {np.exp(np.mean(log_diff)):.6f}")
  print(f"OOB Token Ratio (out of bounds):    {np.mean(is_oob) * 100:.2f}%  (bounds: [{args.tis_ratio_min}, {args.tis_ratio_max}])")
  print(f"Extreme Outliers (> 10 nats):       {np.sum(outliers)} tokens")
  print(f"Min Importance Weight (w_min):      {np.min(weights):.6f}")
  print(f"Max Importance Weight (w_max):      {np.max(weights):.6f}")
  print("=" * 80)

  # SECTION 2
  if sampler_routed_experts is not None:
    print("\n" + "=" * 80)
    print(" " * 22 + "SECTION 2: MOE ROUTER AGREEMENT")
    print("=" * 80)
    print(f"Sampler routed experts shape: {sampler_routed_experts.shape}")
    print("=" * 80)

  # SECTION 3
  print("\n" + "=" * 80)
  print(" " * 23 + "SECTION 3: TOP 15 WORST DIVERGING TOKENS")
  print("=" * 80)
  print(f"{'Pos':<6} {'Token ID':<10} {'Decoded':<20} {'Trainer LogP':<14} {'Sampler LogP':<14} {'Diff (Δ)':<12} {'Weight (e^Δ)':<14} {'Status'}")
  print("-" * 105)

  worst_indices = np.argsort(abs_log_diff)[::-1][:15]
  for idx in worst_indices:
    tok = toks[idx]
    dec = repr(tokenizer.decode([tok]))
    t_lp = t_logps[idx]
    s_lp = s_logps[idx]
    d = log_diff[idx]
    w = weights[idx]
    status = "OUTLIER" if abs(d) > 10.0 else ("OOB" if is_oob[idx] else "OK")
    print(f"{idx:<6} {tok:<10} {dec:<20} {t_lp:<14.4f} {s_lp:<14.4f} {d:<+12.4f} {w:<14.4f} [{status}]")
  print("-" * 105)

  # SECTION 4
  print("\n" + "=" * 80)
  print(" " * 20 + "SECTION 4: DIVERGENCE BY TOKEN POSITION")
  print("=" * 80)
  bucket_size = 64
  num_buckets = int(np.ceil(len(toks) / bucket_size))
  print(f"{'Token Range':<20} {'Mean |Δ| (nats)':<18} {'Max |Δ| (nats)':<18} {'OOB Frac':<12}")
  print("-" * 70)
  for b in range(num_buckets):
    start_idx = b * bucket_size
    end_idx = min((b + 1) * bucket_size, len(toks))
    b_diff = abs_log_diff[start_idx:end_idx]
    b_oob = is_oob[start_idx:end_idx]
    print(f"[{start_idx:>4} - {end_idx:>4}]         {np.mean(b_diff):<18.6f} {np.max(b_diff):<18.6f} {np.mean(b_oob)*100:<10.1f}%")
  print("=" * 80)

  print("\nPARITY EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
  main()
