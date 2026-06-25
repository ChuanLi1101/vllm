# vLLM ROCm CI — Onboarding Quickstart

**Goal of day one:** reproduce *one* red test failure from the nightly, locally, on your own.
That's the bar. If you can do that, you're onboarded — everything else is just repeating it on harder failures.

There is no big knowledge-transfer doc. You pick a test area, start debugging its nightly failures,
and build up knowledge by doing. This page is only here to get you to your first reproduction.

---

## 0. Prereqs (one time)
- Access to a ROCm GPU host (MIxxx) with Docker + `rocminfo` working.
- Buildkite access to the vLLM nightly pipeline.
- `HF_TOKEN` exported for model downloads.

## 1. Find the latest nightly run
- Open the Buildkite **nightly** build for `vllm`.
- Look at the job groups — each maps to a **test area** (see `.buildkite/test_areas/*.yaml`,
  e.g. `quantization.yaml`, `attention.yaml`, `kernels.yaml`, `lora.yaml`, ...).
- Find a **red** job in *your* area. Note two things from the job:
  1. the **commit SHA** of the build, and
  2. the exact **pytest command** the job ran (shown in the job log).

## 2. Pull the nightly docker image
The CI uses one image per commit:
```bash
docker pull rocm/vllm-ci:<BUILDKITE_COMMIT>
```
(`<BUILDKITE_COMMIT>` = the SHA from step 1.)

## 3. Reproduce the failure locally
Easiest path — run the same command the CI runs, via the AMD test wrapper:
```bash
# from the repo root, on the GPU host
export VLLM_TEST_COMMANDS='pytest -v -s quantization/ --ignore quantization/test_blackwell_moe.py'
export BUILDKITE_COMMIT=<the SHA>
bash .buildkite/scripts/hardware_ci/run-amd-test.sh
```
`run-amd-test.sh` pulls the image, mounts your HF cache, applies the ROCm-specific
`--ignore` overrides, and runs the command inside the container exactly like CI does.

> Always pass commands via `VLLM_TEST_COMMANDS` (single-quoted) — passing them as positional
> args mangles inner quotes around `-m`/`-k` expressions.

Prefer to poke around by hand? Drop into the container instead:
```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --network=host --shm-size=16gb \
  --group-add "$(getent group render | cut -d: -f3)" \
  -e HF_TOKEN -v "$HOME/huggingface:/root/.cache/huggingface" \
  rocm/vllm-ci:<BUILDKITE_COMMIT> bash
# then inside:
cd /vllm-workspace/tests && pytest -v -s quantization/ -k <failing_test>
```

## 4. Triage: infra vs. real
Once it fails locally, classify it — this is the most valuable skill:
- **Infra / flaky / third-party** (network, ROCr/rocminfo, image build, download timeout,
  passes on rerun) → not our regression. Note it, file/link a Jira, move on.
- **Real regression** (fails deterministically on the code path) → this is yours to fix → PR.

## 5. Done = onboarded
You reproduced one failure and classified it. Now repeat on the next red in your area,
and start fixing the real ones + adding test cases. Ask in the daily standup / office hour
only when you're genuinely stuck.

---
**Your area is yours.** Watch the nightly for it, keep it green, and fix/PR what's real.
