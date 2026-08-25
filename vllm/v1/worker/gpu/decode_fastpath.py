# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for Model Runner V2 decode fast paths on ROCm TP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def should_wait_output_copy_before_forward(vllm_config: "VllmConfig") -> bool:
    """Whether to drain the async output copy stream before each forward.

    On ROCm TP, overlapping D2H copies on a side stream with the next forward
    pass contends for HBM bandwidth and increases cross-rank allreduce skew.
    """
    if envs.VLLM_V2_WAIT_OUTPUT_COPY is not None:
        return envs.VLLM_V2_WAIT_OUTPUT_COPY

    from vllm.platforms import current_platform

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    return current_platform.is_rocm() and tp_size > 1


def should_defer_async_output_copy(vllm_config: "VllmConfig") -> bool:
    """Whether to start AsyncOutput D2H after postprocess/speculator."""
    if envs.VLLM_V2_DEFER_OUTPUT_COPY is not None:
        return envs.VLLM_V2_DEFER_OUTPUT_COPY

    # Same default as wait: ROCm TP decode benefits from not interleaving copy
    # with the tail of the previous GPU step.
    return should_wait_output_copy_before_forward(vllm_config)
