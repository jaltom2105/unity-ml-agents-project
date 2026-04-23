# Phase 2 — Running the example Unity agents

## Preparing Unity
- Installed Unity Hub and Unity 6.4 (6000.4.3f1)
- Opened ML-Agents example project in Unity 6.4
- Added com.unity.ml-agents package (4.0.3) from disk (didn't create new project)
- Explored and trained three fundamentally different example agents
- Set up and started to learn how a TensorBoard works for real-time training visualization

## The Examples I Explored

### 1. 3DBall — learning to balance
The introductory example where 12 agents learn to balance a ball on their head
by tilting a platform. Simple models but they demonstrate the core training loop

- **Actions:** Only have 2 continuous actions: (tilt X, tilt Z)
- **Observations:** 8 values (ball position, velocity, platform rotation)
- **Result:** Agents reached perfect Mean Reward of 100.000 in ~120,000 steps (~2 minutes)

| Steps | Mean Reward | Observations |
|------|-------------|-------|
| 12,000 | 1.206 | Completely random behavior, ball falling off quickly |
| 60,000 | 8.748 | Starting to learn a little |
| 84,000 | 49.964 | Rapid improvement, learning how to hold the ball for longer |
| 96,000 | 88.586 | Nearly mastered the balance, but still making a few mistakes |
| 120,000 | 100.000 | Perfect score achieved, keeps the ball on head over allotted time |
| 180,000+ | 100.000 | Consistently perfect results |

### 2. Crawler — more complex example of motor control
A spider-like creature learns to crawl forward by coordinating its 20 joint
motors across 4 legs. A lot harder to master than the 3DBall due to the number
of actions and the complexity of coordinated movement over just tilting.

- **Actions:** 20 continuous (joint motors)
- **Key insight:** More actions means it takes much longer to train
- **Observation:** Even at time-scale 1, training is too fast to watch meaningfully or notice
  anything happening
  — best approach is to train at full speed then load the saved model

### 3. SoccerTwos — multi-agent competition
Two teams of agents compete against each other. Uses POCA (Multi-Agent
POsthumous Credit Assignment) algorithm instead of PPO. Both teams learn
simultaneously, creating an automatically scaling difficulty curve.

- **Algorithm:** POCA (config/poca/SoccerTwos.yaml)
- **Key metrics:** Uses an ELO rating instead of simple reward, where each team has their own ELO
- **Key insight:** Mean Reward stays near 0 because rewards only happen at
  goal-scoring moments. ELO is the meaningful metric to watch.
- **Observation:** Needs millions of steps to develop real strategies. Didn't watch the full training because
  it was taking too long

## TensorBoard visualization
Set up TensorBoard to visualize training in real time at http://localhost:6006

### 3DBall Learning Curve Observations
- **Cumulative Reward:** Classic S-curve from 0 → 95, showing slow start,
  rapid growth in strategy, then leveling off once the cubes mastered balancing the ball
- **Episode Length:** Mirrors the reward curve — where longer episodes = ball staying
  balanced longer = more rewards for the cube
- **Policy Loss:** Starts high as agent makes big strategy changes, settles
  down as strategy stabilizes
- **Value Loss:** Peaks during the fastest learning phase, drops off as the reward
  predictions become more accurate

## Key issues that were solved
- Unity project import kept freezing when trying to launch, so I deleted Library folder for a clean import,
  switched from Unity 2022.3 LTS to Unity 6.4
- ML-Agents package version had conflicts, but switching to Unity 6.4 resolved all compatibility issues
- PyTorch repeatedly downgraded to CPU (2.8.0) by ML-Agents installer, so I
  had to reinstall nightly after any pip install operation
- Simulation kept freezing during training, which was caused by CPU-only PyTorch being
  too slow to keep up with Unity. Fixed by restoring the GPU nightly build.
- venv activation in PowerShell needs to use `& "path\to\Activate.ps1"` syntax
- Config file not found errors because Soccer uses config/poca/ not config/ppo/

## Commands references
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
