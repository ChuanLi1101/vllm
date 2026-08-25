#!/usr/bin/env python3
"""MI355X TP8 Kimi-K3 A/B checklist for Model Runner V2 decode fast path.

Run on the same image/config as tp8-pr51653-full-json-traces-20260816.
"""

from __future__ import annotations

BASE_ENV = """
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_V2_WAIT_OUTPUT_COPY=1
export VLLM_V2_DEFER_OUTPUT_COPY=1
"""

SERVE = """
vllm serve moonshotai/Kimi-K3 \\
  --trust-remote-code \\
  --tensor-parallel-size 8 \\
  --gpu-memory-utilization 0.85 \\
  --max-model-len 102400 \\
  --max-num-batched-tokens 4096 \\
  --max-num-seqs 1 \\
  --no-enable-prefix-caching \\
  --compilation-config '{"cudagraph_capture_sizes":[1]}'
"""

BENCH = """
vllm bench serve \\
  --backend vllm \\
  --base-url http://127.0.0.1:8000 \\
  --endpoint /v1/completions \\
  --model moonshotai/Kimi-K3 \\
  --trust-remote-code \\
  --dataset-name random \\
  --random-input-len 100000 \\
  --random-output-len 256 \\
  --random-range-ratio 0.0 \\
  --request-rate inf \\
  --temperature 0 \\
  --ignore-eos \\
  --max-concurrency 1 \\
  --num-prompts 3
"""

PASS_CRITERIA = """
Pass criteria (vs V1 baseline from trace pack):
- TPOT <= 20.0 ms/token (baseline V2=20.9, V1=18.5)
- decode marker mean within +5% of V1 (target ~19.7-20.5 ms)
- optional: re-capture trace and verify decode_1stage mean <= 8.5 us
"""

if __name__ == "__main__":
    print("=== Model Runner V2 ROCm decode fast path A/B ===")
    print(BASE_ENV)
    print(SERVE)
    print(BENCH)
    print(PASS_CRITERIA)
