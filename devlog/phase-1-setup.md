# Phase 1 — Environment Setup

**Date Completed:** April 21, 2026
**Status:** ✅ Complete

## What Was Done
- Installed Python 3.10.11 (64-bit) with PATH configured
- Created Python virtual environment at `C:\Projects\unity-ml-agents-project\venv`
- Installed PyTorch nightly (2.12.0.dev20260408+cu128) with CUDA 12.8 support
- Cloned ML-Agents repo to `C:\Projects\ml-agents`
- Installed `mlagents-envs` and `mlagents` Python packages in editable mode

## Key Issues Solved
- **RTX 5070 Blackwell architecture (sm_120)** is not supported by stable 
  PyTorch — required nightly build with CUDA 12.8
- **ML-Agents dependency conflict** — its installer downgraded torch to 2.8.0,
  requiring a manual reinstall of nightly afterward. The version warning from
  pip can be safely ignored.

## Environment Summary
| Component | Version |
|-----------|---------|
| Python | 3.10.11 |
| PyTorch | 2.12.0.dev20260408+cu128 |
| CUDA | 12.8 |
| mlagents | 1.2.0.dev0 |
| mlagents-envs | 1.2.0.dev0 |
| GPU | NVIDIA GeForce RTX 5070 |
