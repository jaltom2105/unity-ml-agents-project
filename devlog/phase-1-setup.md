# Phase 1 — Setting up Unity and the virtual environment

## What was done
- Installed Python 3.10.11 (64-bit) with PATH configured
- Created Python virtual environment at `C:\Projects\unity-ml-agents-project\venv`
- Installed PyTorch nightly (2.12.0.dev20260408+cu128) with CUDA 12.8 support
- Cloned ML-Agents repo to `C:\Projects\ml-agents`
- Installed `mlagents-envs` and `mlagents` Python packages in editable mode

## Issues that were resolved
- **RTX 5070 Blackwell architecture (sm_120)** is not supported by stable 
  PyTorch, so I needed to install the nightly build with CUDA 12.8
- **ML-Agents dependency conflict** — its installer kept downgrading torch back to 2.8.0,
  which required a manual reinstall of nightly afterward and a constant to be
  created in VS Code. The version warning from pip can be safely ignored.

## Environment summary
| Component | Version |
|-----------|---------|
| Python | 3.10.11 |
| PyTorch | 2.12.0.dev20260408+cu128 |
| CUDA | 12.8 |
| mlagents | 1.2.0.dev0 |
| mlagents-envs | 1.2.0.dev0 |
| GPU | NVIDIA GeForce RTX 5070 |
