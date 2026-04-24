# Phase 3 — Deep Dive into understanding how it works

## Overview
Rather than just continuing to run the examples, I wanted Phase 3 to focus on getting a understanding the mechanics behind reinforcement learning. This meant I was reading actual agent code line by line, learning the reward design philosophy, and running controlled experiments to verify my understanding through real results. In this phase I read the two complete agent scripts for Ball3DAgent.cs and CrawlerAgent.cs. I broke down every parameter in the config YAML file, and ran some controlled experiments that produced real measurable findings.

## 3.1 — How PPO Works

PPO (Proximal Policy Optimization) is the algorithm that powered every training run in this project. Understanding it deeply is essential before building a custom agent.

### The Problem PPO Solves
Before PPO, earlier RL algorithms had a critical flaw — if you updated the policy too aggressively based on new experiences, you could accidentally make it much worse and never recover. PPO solves this with one elegant idea: don't change the policy too much in any single update. The "Proximal" in PPO literally means "nearby" — keep the new policy close to the old one by clipping updates so they can never be too large in a single step.

### What is a Policy?
The agent's strategy is called a policy. It is literally a neural network that takes in observations and outputs actions:

```
Observations (what the agent sees)
        ↓
   Neural Network
        ↓
Actions (what the agent does)
```

For 3DBall specifically:
- Observations in: ball position (x,y,z), ball velocity (x,y,z), platform rotation (x,z) = 8 numbers total
- Actions out: how much to tilt left/right, how much to tilt forward/back = 2 numbers total

### The Training Loop — Step by Step
This exact loop ran thousands of times during every training session:

```
Step 1 — CollectObservations() packages what the agent can see
Step 2 — Neural network receives observations and outputs action values
Step 3 — OnActionReceived() executes those actions in the Unity simulation
Step 4 — Reward is calculated and given to the agent
Step 5 — Experience (observation, action, reward) stored in buffer
Step 6 — Once buffer is full, neural network weights are updated
Step 7 — Repeat from Step 1
```

### The PPO Clipping Mechanism
The key innovation of PPO is the clipping parameter epsilon. With epsilon set to 0.2, the policy ratio between old and new policy is clamped to the range [0.8, 1.2]. This means no single update can change the policy by more than 20%, preventing catastrophic forgetting and ensuring stable learning.

### Why Did 3DBall Learn So Fast?
The 3DBall example went from completely random behavior to perfect balance in about 2 minutes on the RTX 5070. This was due to several factors working together:
- 12 agents training in parallel = 12x the experience per second
- Simple observation space = only 8 numbers to process
- Clear dense reward signal = +0.1 every step the ball stays up, very easy to learn from
- GPU-accelerated neural network updates via CUDA 12.8

---

## 3.2 — Config File Deep Dive

The 3DBall.yaml config file was opened and every single parameter was analyzed in depth. This file controlled every aspect of the training run that produced the learning curve visible in the TensorBoard screenshots.

```yaml
behaviors:
  3DBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 64
      buffer_size: 12000
      learning_rate: 0.0003
      beta: 0.001
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 128
      num_layers: 2
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    keep_checkpoints: 5
    max_steps: 500000
    time_horizon: 1000
    summary_freq: 12000
```

### Parameter Breakdown

**behaviors: 3DBall**
This key maps directly to the Behavior Name set in Unity's Inspector on the agent's Behavior Parameters component. This is how Python knows which config applies to which agent in the scene.

**trainer_type: ppo**
Tells ML-Agents to use Proximal Policy Optimization. Other options include SAC (Soft Actor-Critic) for sample efficiency and POCA for multi-agent scenarios like Soccer.

**batch_size: 64**
How many experiences to use in each neural network update. Think of it like studying 64 flashcards at a time. Too small = noisy unstable learning. Too large = slow updates that miss important patterns.

**buffer_size: 12000**
How many total experiences to collect before doing ANY learning. With 12 agents running in parallel this fills up in about 1000 steps per agent. The agent watches and remembers before it starts adjusting its strategy.

**learning_rate: 0.0003**
How big each step is when updating the neural network weights. This is one of the most important numbers in all of deep learning. Too high = overshoots and never converges. Too low = takes forever to improve. 0.0003 is a well-established safe default for PPO.

**beta: 0.001**
Controls entropy — how much the agent is encouraged to explore random actions rather than always doing what it currently thinks is best. Higher beta = more exploration. Too low and the agent gets stuck in a local optimum early in training, never discovering better strategies.

**epsilon: 0.2**
The PPO clipping parameter. The policy cannot change by more than 20% in any single update. This is the core mechanism that makes PPO stable compared to older algorithms like TRPO.

**lambd: 0.99**
The GAE (Generalized Advantage Estimation) lambda parameter. Controls how far into the future the agent looks when calculating whether an action was good or bad. 0.99 means it considers rewards almost all the way to the end of the episode. Lower values make the agent more shortsighted, higher values introduce more variance.

**num_epoch: 3**
How many times to reuse the same batch of experiences for gradient updates. PPO can safely reuse data a few times unlike older on-policy algorithms. More epochs = more learning per experience collected, but too many risks overfitting to a single batch.

**learning_rate_schedule: linear**
The learning rate starts at 0.0003 and gradually decreases linearly to 0 by max_steps. Like studying hard at the start of a semester then doing lighter review near the final exam.

**normalize: true**
Automatically normalizes all observations to have mean 0 and standard deviation 1 using a running average. Critical for stable training — without this, large observation values can cause exploding or vanishing gradients in the neural network.

**hidden_units: 128**
How many neurons in each hidden layer of the neural network. Combined with num_layers: 2, the agent's complete neural network architecture is:
```
Input Layer (8 neurons) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (128 neurons) → Output Layer (2 neurons)
```
128 is sufficient for a simple task like 3DBall. More complex tasks like Crawler benefit from 256 or 512.

**num_layers: 2**
How many hidden layers in the neural network. More layers allow the network to learn more abstract representations, but also require more data and time to train.

**vis_encode_type: simple**
Only relevant when agents use camera/visual observations. Since 3DBall uses vector observations only, this parameter has no effect here. Would matter for Visual3DBall which learns from pixels.

**gamma: 0.99**
The discount factor for future rewards. A reward received 100 steps from now is worth 0.99^100 = 0.366 of an immediate reward. This makes the agent value immediate rewards slightly more than distant ones, which stabilizes training.

**strength: 1.0**
A multiplier applied to all extrinsic rewards before they are used for learning. Leaving at 1.0 means rewards are used as-is from the C# script.

**keep_checkpoints: 5**
Saves the 5 most recent model checkpoints during training. Allows resuming from a checkpoint or rolling back to an earlier version of the model if training goes wrong.

**max_steps: 500000**
Training automatically stops after this many total environment steps across all agents. With 12 parallel agents, this means each individual agent experiences roughly 41,667 steps.

**time_horizon: 1000**
How many steps the agent collects before calculating advantages and updating. For 3DBall an episode ends when the ball drops, but this caps individual episode contribution at 1000 steps maximum.

**summary_freq: 12000**
How often to write a summary line to PowerShell output and to TensorBoard. This is why the training output updated every 12,000 steps during all training runs.

---

## 3.3 — Reading Agent Code: Ball3DAgent.cs

The complete Ball3DAgent.cs script was read and analyzed line by line. This script is the entire brain of the 3DBall agent — less than 100 lines of C# controls everything observed during training.

### The Four Core Methods
Every ML-Agent ever built uses these same four methods. Understanding them is the complete foundation of building any custom agent including the golf agent in Phase 4.

### Initialize()
```csharp
public override void Initialize()
{
    m_BallRb = ball.GetComponent<Rigidbody>();
    m_ResetParams = Academy.Instance.EnvironmentParameters;
    SetResetParameters();
}
```
Runs exactly once when the agent first spawns. Used to grab references to Unity components that will be needed throughout the agent's lifetime. The Rigidbody reference is needed to read ball velocity in CollectObservations. For the golf agent, Initialize() is where references to the club, ball, ragdoll joints, and any physics components would be cached.

### CollectObservations()
```csharp
public override void CollectObservations(VectorSensor sensor)
{
    if (useVecObs)
    {
        sensor.AddObservation(gameObject.transform.rotation.z); // 1 value — platform tilt Z
        sensor.AddObservation(gameObject.transform.rotation.x); // 1 value — platform tilt X
        sensor.AddObservation(ball.transform.position - gameObject.transform.position); // 3 values — ball position relative to platform
        sensor.AddObservation(m_BallRb.linearVelocity); // 3 values — ball velocity
    }
}
```
Called every single step. The agent can ONLY know what is explicitly added here using sensor.AddObservation() — nothing else in the Unity scene is visible to the neural network. The total of 8 values here exactly matches the Space Size: 8 visible in Unity's Inspector on the Behavior Parameters component.

This is one of the most critical design decisions when building a custom agent. Too few observations and the agent lacks the information needed to learn. Too many irrelevant observations and training slows down with noise the network has to learn to ignore.

### OnActionReceived()
```csharp
public override void OnActionReceived(ActionBuffers actionBuffers)
{
    var actionZ = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
    var actionX = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);

    if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
        (gameObject.transform.rotation.z > -0.25f && actionZ < 0f))
    {
        gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
    }

    if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
        (gameObject.transform.rotation.x > -0.25f && actionX < 0f))
    {
        gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
    }

    if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
        Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
        Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
    {
        SetReward(-1f);
        EndEpisode();
    }
    else
    {
        SetReward(0.1f);
    }
}
```
Called every step after the neural network processes the observations and outputs action values. Two things happen here: actions are executed in the simulation and rewards are assigned.

The neural network outputs two numbers between -1 and 1. Multiplying by 2f scales them to a rotation speed between -2 and 2 degrees per step. Mathf.Clamp ensures the network output can never go outside the valid range regardless of what the neural network outputs.

The rotation safety checks prevent the platform from tilting more than approximately 14 degrees in any direction. The agent has full freedom within that range but cannot tip the platform completely over.

The reward function is the most important part:
- Ball fell off the platform → SetReward(-1f) and EndEpisode() — punish the failure and immediately reset
- Ball still on platform → SetReward(0.1f) — small reward every single step

The +0.1 per step is why the agent learned to keep the ball on as long as possible — every extra step alive earns more cumulative reward. This is called a dense reward signal because feedback is given every single step rather than only at the end. Dense rewards are why 3DBall trains so quickly.

### OnEpisodeBegin()
```csharp
public override void OnEpisodeBegin()
{
    gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
    gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
    gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
    m_BallRb.linearVelocity = new Vector3(0f, 0f, 0f);
    ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f))
        + gameObject.transform.position;
    SetResetParameters();
}
```
Called every time an episode ends — either from EndEpisode() being called or from the Max Step limit being reached. The randomization here is absolutely critical. Without it the agent would memorize one specific starting condition instead of learning a general balancing strategy.

If the platform always started perfectly flat with the ball always in the same position, the agent would learn a fixed sequence of movements that only works for that exact starting state. With randomization it is forced to learn the underlying principle of balance that works from any starting position. This concept is called domain randomization and it is essential for producing robust trained models.

### Heuristic()
```csharp
public override void Heuristic(in ActionBuffers actionsOut)
{
    var continuousActionsOut = actionsOut.ContinuousActions;
    continuousActionsOut[0] = -Input.GetAxis("Horizontal");
    continuousActionsOut[1] = Input.GetAxis("Vertical");
}
```
Allows manual keyboard control of the agent using arrow keys instead of the neural network. This is used purely for testing and debugging — by switching Behavior Type to Heuristic in the Inspector, the environment can be manually tested to verify it works correctly before spending hours on training. Always implement this method for custom agents.

---

## 3.4 — Reading Agent Code: CrawlerAgent.cs

The complete CrawlerAgent.cs was read and compared against Ball3DAgent.cs to understand how the same four methods scale to a dramatically more complex problem. Crawler uses 20 continuous actions compared to 3DBall's 2, and introduces several advanced techniques not present in 3DBall.

### The OrientationCube — A Key Engineering Concept
```csharp
m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
```
Crawler introduces an invisible helper object called the OrientationCube that is not present in 3DBall. This object always rotates to point toward the target the crawler is walking to. All observations are expressed relative to this cube rather than in world space.

This is critically important. If observations were in world space, the agent would learn "move in the positive Z direction" rather than "move toward the target." By expressing everything relative to the OrientationCube, the agent learns the more general and transferable concept of "move in whatever direction the target is." This is called using a local reference frame and it is a fundamental design principle for any agent that needs to navigate or orient toward something — directly applicable to the golf agent.

### Observations — Local Reference Frame
```csharp
public override void CollectObservations(VectorSensor sensor)
{
    var cubeForward = m_OrientationCube.transform.forward;

    var velGoal = cubeForward * TargetWalkingSpeed;
    var avgVel = GetAvgVelocity();

    // How far off is current speed from target speed
    sensor.AddObservation(Vector3.Distance(velGoal, avgVel));

    // Body velocity expressed relative to orientation cube — not world space
    sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));

    // Target velocity expressed relative to orientation cube — not world space
    sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));

    // How much is body rotation misaligned with target direction
    sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));

    // Where is the target relative to orientation cube — not world space
    sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

    // Raycast downward to measure height above ground
    RaycastHit hit;
    float maxRaycastDist = 10;
    if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist))
    {
        sensor.AddObservation(hit.distance / maxRaycastDist);
    }
    else
        sensor.AddObservation(1);

    // For every body part — is it touching the ground, and how much joint force is it using
    foreach (var bodyPart in m_JdController.bodyPartsList)
    {
        CollectObservationBodyPart(bodyPart, sensor);
    }
}

public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
{
    sensor.AddObservation(bp.groundContact.touchingGround);
    if (bp.rb.transform != body)
    {
        sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
    }
}
```
InverseTransformDirection and InverseTransformPoint convert world-space vectors into local-space vectors relative to the OrientationCube. The agent perceives the world from its own oriented perspective rather than from a fixed global coordinate system.

### Actions — 20 Joint Motors
Where 3DBall had 2 actions, Crawler has 20 — one set of rotation and strength values per joint:
```csharp
public override void OnActionReceived(ActionBuffers actionBuffers)
{
    var bpDict = m_JdController.bodyPartsDict;
    var continuousActions = actionBuffers.ContinuousActions;
    var i = -1;

    // Upper legs can rotate in two axes
    bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
    bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
    bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
    bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

    // Lower legs (knees) can only rotate in one axis
    bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
    bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
    bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
    bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);

    // Each joint also has an independent strength value
    bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
    bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
    bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
    bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
    bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
    bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
    bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
    bpDict[leg3Lower].SetJointStrength(continuousActions[++i]);
}
```
Each upper leg needs two rotation values because hip joints can rotate in two axes. Each lower leg (knee) only needs one rotation value because knees only bend in one direction. Each joint additionally has a strength value — the agent controls not just the target direction but how much force to apply to reach it. This is directly relevant to the golf agent which will need to control joint torque magnitude and timing to produce a proper swing.

### The Crawler Reward Function — Shaped Rewards and Reward Gating
The reward function in Crawler is the most significant departure from 3DBall and contains the most important lessons for designing custom reward functions:

```csharp
void FixedUpdate()
{
    UpdateOrientationObjects();

    // Reward component 1: how well does current speed match target speed
    var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());

    // Reward component 2: how well is body facing toward the target
    var lookAtTargetReward = (Vector3.Dot(cubeForward, body.forward) + 1) * .5F;

    // Multiply — both must be true simultaneously to earn reward
    AddReward(matchSpeedReward * lookAtTargetReward);
}

public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
{
    var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);
    return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
}
```

**matchSpeedReward** is a sigmoid-shaped curve that returns 1.0 when the crawler's actual speed perfectly matches the target walking speed, and smoothly approaches 0 as speed deviates. This is a shaped reward — it always gives partial credit for being close to the goal. This gives the neural network a gradient to follow even when performance is far from perfect, which is why shaped rewards work better than binary pass/fail for complex tasks.

**lookAtTargetReward** uses the dot product:
```csharp
(Vector3.Dot(cubeForward, body.forward) + 1) * .5F
```
The dot product of two unit vectors returns 1.0 when they point in exactly the same direction, 0.0 when perpendicular, and -1.0 when pointing in exactly opposite directions. Adding 1 and multiplying by 0.5 maps this to a clean 0-1 range. The result is 1.0 when the crawler perfectly faces the target and 0.0 when it faces completely away.

**Multiplying the two components** is the key design decision:
```csharp
AddReward(matchSpeedReward * lookAtTargetReward);
```
If either value approaches zero, the entire reward collapses toward zero. The agent must satisfy BOTH conditions simultaneously to earn meaningful reward. If it moves fast in the wrong direction lookAtTargetReward is near zero so total reward is near zero. If it faces the target perfectly but stands still matchSpeedReward is near zero so total reward is near zero. Only moving fast in the right direction earns full reward.

This is called reward gating — one reward gates whether the other pays out. It is far more effective than adding the two components which would allow the agent to earn partial reward by doing either condition without the other.

Note that the reward is given in FixedUpdate rather than OnActionReceived — this is valid and means reward is calculated every physics frame independently of when the neural network processes observations.

---

## 3.5 — Experiments

### Experiment 1 — Network Size vs Performance

**Hypothesis:** Larger networks should learn faster due to greater representational capacity.

Three config files were created in C:\Projects\ml-agents\config\ppo\, identical in every parameter except hidden_units:
- 3DBall_64units.yaml — hidden_units: 64
- 3DBall_128units_baseline.yaml — hidden_units: 128 (default)
- 3DBall_256units.yaml — hidden_units: 256

Each was trained to 204,000 steps and compared on first step reaching Mean Reward 100.000.

**Results:**

| Run | Hidden Units | First Perfect Score | Notes |
|-----|-------------|-------------------|-------|
| A | 64 | Step 132,000 | Small network |
| B | 128 (baseline) | Step 132,000 | Default config |
| C | 256 | Step 132,000 | Large network |

**Finding:** All three networks hit perfect score at exactly the same step. For a simple task like 3DBall, network capacity is not the bottleneck. Even the smallest 64-unit network has more than sufficient capacity to learn 2 actions from 8 observations. Increasing to 256 units provided zero measurable benefit.

This demonstrates the concept of model capacity vs task complexity. The network only needs to be as large as the task requires. Where network size would matter is on genuinely complex tasks like Crawler with 20 actions and complex multi-joint coordination — that is where 256 units would likely outperform 64.

**TensorBoard Screenshot:** screenshots/Network_size_experiment_3DBALL_T-board.png
Three nearly identical overlaid S-curves visually confirming the result. All three reward curves, episode length curves, policy loss curves, and value loss curves are indistinguishable from each other.

---

### Experiment 2 — Reward Scaling

**Hypothesis:** Reducing the ratio between penalty and survival reward would destabilize learning by weakening the failure signal.

The survival reward in Ball3DAgent.cs was changed from 0.1f to 1.0f while the failure penalty remained at -1.0f:

```csharp
// Original — 10:1 penalty to reward ratio
SetReward(0.1f);

// Modified — 1:1 penalty to reward ratio
SetReward(1.0f);
```

**Results:**

| Version | Survival Reward | Failure Penalty | Ratio | First Perfect | Stability |
|---------|----------------|-----------------|-------|--------------|-----------|
| Original | +0.1 | -1.0 | 10:1 | Step 132,000 | Stable |
| Modified | +1.0 | -1.0 | 1:1 | Step 156,000 | Unstable |

The modified version showed characteristic instability — reward climbed to 95.642 at step 120,000 then dropped back down to 79.363 at step 132,000 before eventually stabilizing at 100 at step 156,000. The original version never exhibited this kind of regression once it started climbing.

**Finding:** The ratio between rewards matters as much as the absolute values. The original 10:1 penalty-to-reward ratio created a strong failure signal that dominated the agent's learning. Changing to a 1:1 ratio made failure feel relatively minor compared to the accumulating survival rewards, causing the agent to underweight failures and produce unstable oscillating learning. This concept is called reward scaling and it is a fundamental consideration when designing any custom reward function.

---

### Experiment 3 — Reward Hacking

**Hypothesis:** Removing all penalties and episode resets would cause the agent to find an unintended shortcut to maximize reward without actually learning to balance.

The failure condition in Ball3DAgent.cs was modified to comment out both the penalty and the EndEpisode() call, replacing failure with the same reward as survival:

```csharp
// Original
if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
    Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
    Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
{
    SetReward(-1f);
    EndEpisode();
}
else
{
    SetReward(0.1f);
}

// Modified — penalty and reset removed, identical reward in both branches
if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
    Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
    Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
{
    // SetReward(-1f);
    // EndEpisode();
    SetReward(0.1f); // identical to surviving
}
else
{
    SetReward(0.1f);
}
```

**Result:** Mean reward hit 100 instantly. Not because the agents learned to balance — but because they did absolutely nothing. With no EndEpisode() call, episodes ran the full Max Step limit of 5000 steps. Since both ball-on-platform and ball-on-ground gave identical +0.1 rewards, the neural network had no way to distinguish between success and failure states. The agents were visually observed in Unity making zero attempt to balance — balls sitting stationary on the ground while platforms made no movement whatsoever. The agent had discovered that standing completely still was the optimal strategy for maximizing the reward signal as defined.

This is textbook reward hacking — the agent found a completely unintended path to maximum reward that entirely bypassed the intended task.

**Finding:** This experiment demonstrated the single most important principle in all of reinforcement learning:

> An agent does not learn what you want it to learn. It learns whatever maximizes the number you give it. If that number can be maximized by doing nothing, it will do nothing. If it can be maximized by doing something weird and unintended, it will do that weird thing.

Real world examples of reward hacking in research:
- A boat racing agent learned to drive in circles collecting powerups instead of finishing the race
- A robot arm learned to position itself between the camera and the object so it looked like it was grasping without actually touching it
- A simulated robot learned to grow extremely tall and fall over to reach the finish line faster than actually walking

This directly informs Phase 4 golf agent design. Every reward component must be designed such that the only way to maximize it is to actually perform the intended behavior. Naive rewards like "ball moves forward" can be hacked by nudging the ball back and forth. "Ball reaches hole" cannot be hacked but is so sparse the agent may never receive a signal. The golf reward function must be carefully layered to be both learnable and unhackable.

---

## 3.6 — Key Concepts Mastered

| Concept | Definition |
|---------|-----------|
| Policy | The neural network — maps observations to actions |
| PPO clipping | Prevents policy from changing more than epsilon per update, ensuring stability |
| Dense reward | Feedback given every single step — trains much faster |
| Sparse reward | Feedback only given at success or failure — much harder to learn from |
| Shaped reward | Continuous reward using sigmoid or similar curve — gives partial credit for being close |
| Reward hacking | Agent finds an unintended way to maximize reward without solving the task |
| Reward scaling | The ratio between reward values matters as much as the absolute values |
| Reward gating | Multiplying reward components so both must be satisfied simultaneously |
| Domain randomization | Randomizing starting conditions each episode to prevent overfitting |
| Local reference frame | Expressing observations relative to agent orientation rather than world space |
| Overfitting | Memorizing one specific scenario instead of learning a general transferable skill |
| Model capacity | Network must be large enough for the task but larger is not always better |
| Dot product | Mathematical tool measuring directional alignment between two vectors — returns 1 when parallel, 0 when perpendicular, -1 when opposite |
| InverseTransformDirection | Unity method converting a world-space direction into local-space relative to a transform |
| Advantage | How much better or worse an action was compared to what the agent expected |
| Buffer | Collection of experiences gathered before any learning update occurs |
| Batch | Subset of buffer used for a single neural network gradient update |

---

## Screenshots
- screenshots/3DBall_TensorBoard_4-21-26.png — Single baseline run showing the complete S-curve learning progression from 0 to 100 reward
- screenshots/Network_size_experiment_3DBALL_T-board.png — All three network size experiment runs overlaid on the same TensorBoard graph, showing three nearly identical S-curves confirming network size had no effect on this task

## Commands Used This Phase
```powershell
# Activate virtual environment
& "C:\Projects\unity-ml-agents-project\venv\Scripts\Activate.ps1"

# Network size experiment
mlagents-learn config/ppo/3DBall_64units.yaml --run-id=3DBall-64units --time-scale=20
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall-128units --time-scale=20
mlagents-learn config/ppo/3DBall_256units.yaml --run-id=3DBall-256units --time-scale=20

# Reward scaling experiment
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall-reward-experiment --time-scale=20

# Reward hacking experiment
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall-no-penalty-real --time-scale=20

# TensorBoard visualization
tensorboard --logdir results

# Verify GPU after any pip install
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Reinstall nightly PyTorch if downgraded by pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```

## Next Phase
Phase 4 — Custom Agent: Ragdoll golfer learning to swing a club and hit a ball into a hole
