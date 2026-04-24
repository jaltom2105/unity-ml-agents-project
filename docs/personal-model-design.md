# Phase 4 — Custom Agent Design Document
**Golf Swing Ragdoll Agent**
Space to plan and develop the design of the agent

---

## Project Vision
A humanoid ragdoll that learns to swing a golf club and make 
solid contact with a golf ball using reinforcement learning. The agent 
starts with no knowledge of what a golf swing is and gradually discovers 
the correct motion through trial and error guided by a designed 
reward function.

The project progresses through three training stages:
- Stage 1 — Learn the swing motion (no ball)
- Stage 2 — Make contact with the ball
- Stage 3 — Hit the ball with power and direction

---

## Agent body design

### Structure
A minimal humanoid ragdoll with enough joints to produce a realistic 
golf swing. Legs provide stability, upper body produces the swing.

```
         [Head]
           |
        [Torso]  ← primary rotation source for swing
       /        \
  [L_Shoulder]  [R_Shoulder]
       |               |
  [L_Elbow]       [R_Elbow]
       |               |
  [L_Hand]        [R_Hand]
         \          /
          [Golf Club]  ← attached to BOTH hands (because thats how one naturally swings, not just holding in one hand)
```

### Joint Constraints
All joints are constrained to realistic human ranges of motion:

| Joint | Axes of motion | Range of motion |
|-------|------|-------|
| Torso | Y rotation only | -120° to +120° |
| Left Shoulder | X and Y | X: -60° to +180°, Y: -90° to +90° |
| Right Shoulder | X and Y | X: -60° to +180°, Y: -90° to +90° |
| Left Elbow | X only | 0° to +145° |
| Right Elbow | X only | 0° to +145° |
| Left Hip | X and Y | X: -30° to +100°, Y: -45° to +45° |
| Right Hip | X and Y | X: -30° to +100°, Y: -45° to +45° |
| Left Knee | X only | 0° to +135° |
| Right Knee | X only | 0° to +135° |

### Club attachment
The golf club is attached rigidly onto both hands with a standard right-handed grip:
- Left hand: higher on the grip
- Right hand: lower on the grip
- Club head has a trigger collider for ball contact detection

---

## Observations (there will be 26 total)

| Observation | Values | Why we need this observation |
|-------------|--------|-----|
| Torso rotation (quaternion) | 4 | Core swing rotation |
| Torso angular velocity | 3 | How fast the torso is rotating |
| Torso position relative to ball | 3 | Is agent standing over the ball |
| Hip rotation | 2 | Lower body position |
| Right shoulder rotation | 2 | Primary swing arm |
| Right elbow rotation | 1 | Club extension |
| Left shoulder rotation | 2 | Lead arm |
| Left elbow rotation | 1 | Lead arm extension |
| Club head velocity | 3 | Speed at impact — most important |
| Ball position relative to feet | 3 | Where is the ball |
| Ball velocity | 3 | Zero until hit, then tells us result |

---

## Actions (14 total, all continuous)

| Action | Description |
|--------|-------------|
| Right Shoulder X | Arm raise/lower |
| Right Shoulder Y | Arm rotate inward/outward |
| Right Elbow | Arm bend/extend |
| Left Shoulder X | Lead arm raise/lower |
| Left Shoulder Y | Lead arm rotate |
| Left Elbow | Lead arm bend/extend |
| Torso Y | Hip/torso rotation — core of swing |
| Hip rotation | Lower body stability |
| Joint strengths ×6 | Force applied to each joint |

---

## The reward functions (the juicy stuff :D)

### Active Components

**Ball Contact Bonus**
```csharp
reward += 1.0f;  // when club collider contacts ball
```
Clear signal that the most important event occurred, making it so the 
agent is encouraged to hit the ball.

**Club Head Speed at Impact**
```csharp
reward += clubHeadSpeed * 0.1f;  // at moment of contact
```
Encourages building swing speed before impact, bringing the club back 
then forward towards the ball.

**Ball Distance Traveled**
```csharp
reward += ballDistanceTraveled * 0.5f;  // measured 2 seconds after contact
```
Encourages hitting the ball with power and direction not just grazing it.

**Swing Plane Reward (shaped reward — the core of good form)**

This reward system uses cross products and dot products to mathematically 
define a perfect swing plane and reward the agent for following it.

Step 1 — Define the swing plane using cross products:
```csharp
// Vector from shoulder line to club face
Vector3 shoulderToClub = clubHead.position - shoulderLine.position;

// Vector from shoulder line to right shoulder  
Vector3 shoulderToRightShoulder = rightShoulder.position - shoulderLine.position;

// Cross product defines the swing plane normal — the axis to rotate around
Vector3 swingPlaneNormal = Vector3.Cross(shoulderToClub, shoulderToRightShoulder);

// Cross product again gives exact direction club should be moving
Vector3 targetClubDirection = Vector3.Cross(swingPlaneNormal, shoulderToClub);
```

Step 2 — Reward club velocity alignment with target direction:
```csharp
// Dot product: 1.0 when perfectly aligned, 0 when perpendicular, negative when wrong
float swingAlignmentReward = Vector3.Dot(
    clubHeadRigidbody.linearVelocity.normalized, 
    targetClubDirection.normalized
);
reward += swingAlignmentReward;
```

Step 3 — Punish deviation from the swing plane:
```csharp
// How far has the club drifted off the ideal swing plane
float deviationFromPlane = Vector3.Distance(
    clubHead.position, 
    swingPlaneLine  // projected position on the plane
);
reward -= deviationFromPlane * 0.1f;
```

Step 4 — Reverse and square the reward on the downswing:
```csharp
// Once club passes the ball angle, reverse reward direction and square it
// Squaring makes high-speed downswing much more rewarding than slow swing
if (clubAngleRelativeToBall > impactThreshold)
{
    float downswingReward = Vector3.Dot(
        clubHeadRigidbody.linearVelocity.normalized, 
        -targetClubDirection.normalized
    );
    reward += Mathf.Pow(downswingReward, 2);
}
```

The squaring of the downswing reward is critical — it creates a non-linear 
incentive where committing fully to the downswing earns dramatically more 
reward than a half-hearted swing, naturally teaching the agent to accelerate 
through impact.
---

## Episode structure

```
OnEpisodeBegin:
├── Reset ragdoll to standing address position
│   └── Small random variation in stance (domain randomization)
├── Place golf ball at fixed tee position
│   └── No random offset — golf ball position is consistent like a real tee
└── Reset club to address position

During Episode:
├── Agent swings (or tries to)
├── Contact detected → measure club head speed, trigger distance timer
├── 2 seconds after contact → measure ball distance, end episode
└── Max 2000 steps → end episode

No falling penalty — agent is free to fall over, it just won't earn reward
```

---

## Training curriculum

### Stage 1 — Learn to Swing
- No ball in the scene
- Reward: club head speed + swing form only
- Goal: agent discovers the basic motion of a golf swing
- Config: config/ppo/Golf_stage1.yaml

### Stage 2 — Make Contact  
- Ball added at fixed tee position
- Reward: contact bonus + club head speed at impact
- Goal: agent learns to aim the swing at the ball
- Config: config/ppo/Golf_stage2.yaml

### Stage 3 — Hit it Forward
- Full reward function active
- Reward: all components including ball distance
- Goal: agent learns to hit with power and direction
- Config: config/ppo/Golf_stage3.yaml

---

## Unity Scene Structure

```
GolfScene/
├── GolfAgent/
│   ├── Torso (Rigidbody + Collider)
│   ├── Head (Rigidbody + Collider)
│   ├── LeftUpperArm (Rigidbody + ConfigurableJoint)
│   ├── LeftForeArm (Rigidbody + ConfigurableJoint)
│   ├── LeftHand (Rigidbody + ConfigurableJoint)
│   ├── RightUpperArm (Rigidbody + ConfigurableJoint)
│   ├── RightForeArm (Rigidbody + ConfigurableJoint)
│   ├── RightHand (Rigidbody + ConfigurableJoint)
│   ├── LeftThigh (Rigidbody + ConfigurableJoint)
│   ├── LeftShin (Rigidbody + ConfigurableJoint)
│   ├── LeftFoot (Rigidbody + ConfigurableJoint)
│   ├── RightThigh (Rigidbody + ConfigurableJoint)
│   ├── RightShin (Rigidbody + ConfigurableJoint)
│   ├── RightFoot (Rigidbody + ConfigurableJoint)
│   └── GolfClub/
│       ├── Shaft (Box Collider)
│       └── ClubHead (Box Collider + Trigger for contact detection)
├── GolfBall (Sphere + Rigidbody + Physics Material)
├── Ground (Plane)
└── Main Camera
```

---

## C# Scripts Required

| Script | Purpose |
|--------|---------|
| GolfAgent.cs | Main agent — observations, actions, rewards |
| GolfBallController.cs | Detects contact, measures distance traveled |
| JointDriveController.cs | Reuse from Crawler — manages joint forces |
| GroundContact.cs | Reuse from Crawler — detects ground touching |

---

## Open Questions / To Be Decided
- [ ] Exact visual style of the ragdoll character
- [ ] Whether to add a target/flag in Stage 3 for directional reward
- [ ] Whether to eventually progress to a putting mechanic
- [ ] Color scheme and visual personality of the character

---

## Development Log
- April 23, 2026 — Design document created
