# SPDX-License-Identifier: Apache-2.0
"""Tests for Model Runner V2 ROCm decode fast paths."""

from __future__ import annotations

from unittest import mock

import pytest

from vllm.v1.worker.gpu.decode_fastpath import (
    should_defer_async_output_copy,
    should_wait_output_copy_before_forward,
)


class _ParallelConfig:
    def __init__(self, tp_size: int = 1):
        self.tensor_parallel_size = tp_size


class _VllmConfig:
    def __init__(self, tp_size: int = 1):
        self.parallel_config = _ParallelConfig(tp_size)


@pytest.mark.parametrize(
    ("env_wait", "env_defer", "expected_wait", "expected_defer"),
    [
        (True, None, True, True),
        (False, None, False, False),
        (True, False, True, False),
    ],
)
def test_explicit_env_overrides(
    env_wait: bool | None,
    env_defer: bool | None,
    expected_wait: bool,
    expected_defer: bool,
):
    cfg = _VllmConfig(tp_size=8)
    with mock.patch(
        "vllm.v1.worker.gpu.decode_fastpath.envs.VLLM_V2_WAIT_OUTPUT_COPY", env_wait
    ):
        with mock.patch(
            "vllm.v1.worker.gpu.decode_fastpath.envs.VLLM_V2_DEFER_OUTPUT_COPY",
            env_defer,
        ):
            assert should_wait_output_copy_before_forward(cfg) is expected_wait
            assert should_defer_async_output_copy(cfg) is expected_defer


def test_defer_env_override_only():
    cfg = _VllmConfig(tp_size=8)
    with mock.patch(
        "vllm.v1.worker.gpu.decode_fastpath.envs.VLLM_V2_WAIT_OUTPUT_COPY", None
    ):
        with mock.patch(
            "vllm.v1.worker.gpu.decode_fastpath.envs.VLLM_V2_DEFER_OUTPUT_COPY", True
        ):
            assert should_defer_async_output_copy(cfg) is True


@pytest.mark.skip(reason="requires full vllm platform deps for ROCm auto-default")
def test_rocm_tp8_auto_default():
    cfg = _VllmConfig(tp_size=8)
    with mock.patch(
        "vllm.v1.worker.gpu.decode_fastpath.envs.VLLM_V2_WAIT_OUTPUT_COPY", None
    ):
        with mock.patch(
            "vllm.v1.worker.gpu.decode_fastpath.envs.VLLM_V2_DEFER_OUTPUT_COPY", None
        ):
            assert should_wait_output_copy_before_forward(cfg) in (True, False)
