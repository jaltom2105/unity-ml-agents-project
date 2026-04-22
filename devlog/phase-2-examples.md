# Phase 2 — Running Example Agents

## What Was Done
- Installed Unity Hub and Unity 6.4 (6000.4.3f1)
- Opened ML-Agents example project in Unity 6.4
- Added com.unity.ml-agents package (4.0.3) from disk
- Explored and trained three fundamentally different example agents
- Set up and started to learn how a TensorBoard works for real-time training visualization

## The Examples I Explored

### 1. 3DBall — Balance Control
The introductory example. 12 agents learn to balance a ball on their head
by tilting a platform. Simple but demonstrates the core training loop perfectly.

- **Actions:** 2 continuous (tilt X, tilt Z)
- **Observations:** 8 values (ball position, velocity, platform rotation)
- **Result:** Agents reached perfect Mean Reward of 100.000 in ~120,000 steps (~2 minutes)

| Step | Mean Reward | Notes |
|------|-------------|-------|
| 12,000 | 1.206 | Completely random behavior |
| 60,000 | 8.748 | Starting to learn |
| 84,000 | 49.964 | Rapid improvement |
| 96,000 | 88.586 | Nearly mastered |
| 120,000 | 100.000 | Perfect score achieved |
| 180,000+ | 100.000 | Consistently perfect |

### 2. Crawler — Complex Motor Control
A spider-like creature learns to crawl forward by coordinating 20 joint
motors across 4 legs. Exponentially harder than 3DBall due to the number
of actions and the complexity of coordinated movement.

- **Actions:** 20 continuous (joint motors)
- **Key insight:** More actions = much longer training time needed
- **Observation:** Even at time-scale 1, training is too fast to watch meaningfully
  — best approach is to train at full speed then load the saved model for inference

### 3. SoccerTwos — Multi-Agent Competition
Two teams of agents compete against each other. Uses POCA (Multi-Agent
POsthumous Credit Assignment) algorithm instead of PPO. Both teams learn
simultaneously, creating an automatically scaling difficulty curve.

- **Algorithm:** POCA (config/poca/SoccerTwos.yaml)
- **Key metrics:** ELO rating instead of simple reward
- **Key insight:** Mean Reward stays near 0 because rewards only happen at
  goal-scoring moments. ELO is the meaningful metric to watch.
- **Observation:** Needs millions of steps to develop real strategies. Didn't watch the full training.

## TensorBoard Visualization
Set up TensorBoard to visualize training in real time at http://localhost:6006

### 3DBall Learning Curve Observations
- **Cumulative Reward:** Classic S-curve from 0 → 95, showing slow start,
  rapid improvement, then leveling off at mastery
- **Episode Length:** Mirrors reward curve — longer episodes = ball staying
  balanced longer
- **Policy Loss:** Starts high as agent makes big strategy changes, settles
  down as strategy stabilizes
- **Value Loss:** Peaks during fastest learning phase, drops as reward
  predictions become accurate

## Key Issues Solved
- Unity project import freezing → deleted Library folder for clean import,
  switched from Unity 2022.3 LTS to Unity 6.4
- ML-Agents package version conflicts → Unity 6.4 resolved all compatibility issues
- PyTorch repeatedly downgraded to CPU (2.8.0) by ML-Agents installer →
  must reinstall nightly after any pip install operation
- Simulation freezing during training → caused by CPU-only PyTorch being
  too slow to keep up with Unity. Fixed by restoring GPU nightly build.
- venv activation in PowerShell → use `& "path\to\Activate.ps1"` syntax
- Config file not found errors → Soccer uses config/poca/ not config/ppo/

## Commands Reference
```powershell
# Activate venv
& "C:\Projects\unity-ml-agents-project\venv\Scripts\Activate.ps1"

# Train 3DBall
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall-run1 --time-scale=20

# Train Crawler
mlagents-learn config/ppo/Crawler.yaml --run-id=Crawler-run1 --time-scale=5

# Train Soccer
mlagents-learn config/poca/SoccerTwos.yaml --run-id=Soccer-run1 --time-scale=3

# Start TensorBoard
tensorboard --logdir results

# Verify GPU
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Reinstall nightly PyTorch if downgraded
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```
