# 🧊 Q-Learning on FrozenLake-v1

> **Experiment 2 — Implementation of Q-Learning:**  
> Solve the FrozenLake environment using Q-Learning, experimenting with different Learning Rates, Discount Factors, and Exploration Strategies using the **Epsilon-Greedy** method.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [What is Reinforcement Learning?](#-what-is-reinforcement-learning)
3. [Q-Learning Algorithm](#-q-learning-algorithm)
4. [The FrozenLake Environment](#-the-frozenlake-environment)
5. [Key Concepts & Parameters](#-key-concepts--parameters)
6. [Code Walkthrough — Step by Step](#-code-walkthrough--step-by-step)
7. [How the Q-Table Works](#-how-the-q-table-works)
8. [Epsilon-Greedy Strategy](#-epsilon-greedy-strategy)
9. [Training Loop Explained](#-training-loop-explained)
10. [Libraries & Technologies Used](#-libraries--technologies-used)
11. [Running the Notebook](#-running-the-notebook)
12. [Results & Output](#-results--output)
13. [Key Takeaways](#-key-takeaways)

---

## 🗂️ Project Overview

This project implements **Q-Learning** — a model-free Reinforcement Learning algorithm — to train an agent to navigate the `FrozenLake-v1` environment provided by [OpenAI Gymnasium](https://gymnasium.farama.org/).

The agent starts at position `S` (Start) on a frozen grid and must reach position `G` (Goal) while avoiding `H` (Holes). The ice is **slippery**, meaning the agent may slide in an unintended direction, making this a stochastic problem.

### 🎯 Objectives
- Understand the fundamentals of Q-Learning
- Train an agent from scratch using a Q-table
- Observe how different hyperparameters (α, γ, ε) affect learning behaviour
- Implement epsilon decay to shift from exploration to exploitation over time

---

## 🤖 What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent takes **actions**, receives **rewards** (or penalties), and its goal is to **maximize cumulative reward** over time.

```
  Agent ──action──► Environment
    ▲                   │
    └─── reward + state ◄┘
```

### Core RL Components

| Component | Description |
|-----------|-------------|
| **Agent** | The learner/decision-maker (our program) |
| **Environment** | The world the agent interacts with (FrozenLake) |
| **State (s)** | The current situation of the agent (position on the grid) |
| **Action (a)** | What the agent can do (Left, Right, Up, Down) |
| **Reward (r)** | The feedback signal after each action |
| **Policy (π)** | The strategy mapping states → actions |
| **Value Function** | Expected cumulative reward from a given state |

---

## 🧠 Q-Learning Algorithm

**Q-Learning** is a **model-free, off-policy** RL algorithm. It learns the optimal action-selection policy by iteratively updating a table of Q-values — the quality of taking a specific action in a specific state.

### The Bellman Equation (Q-Learning Update Rule)

```
Q(s, a) ← Q(s, a) + α × [ r + γ × max_a' Q(s', a') − Q(s, a) ]
```

Where:
- `Q(s, a)` — Current Q-value for state `s` and action `a`
- `α` (alpha) — **Learning Rate**: how much we update the old estimate
- `r` — **Reward** received after taking action `a` from state `s`
- `γ` (gamma) — **Discount Factor**: how much future rewards are valued
- `s'` — **Next State** after the action
- `max_a' Q(s', a')` — Maximum Q-value achievable from the next state
- `Q(s, a)` (last term) — **Temporal Difference Error (TD Error)**

### Why "Model-Free"?
Q-Learning doesn't need a model of the environment (i.e., it doesn't need to know transition probabilities P(s'|s,a)). The agent learns purely from **trial and error**.

### Why "Off-Policy"?
The agent learns the optimal policy while potentially following a **different (exploratory) policy**. It updates Q-values using the *greedy* (best) action from the next state, even if it didn't take that action.

---

## 🧊 The FrozenLake Environment

### Grid Layout (4×4)

```
S  F  F  F       S = Start (safe)
F  H  F  H       F = Frozen (safe tile)
F  F  F  H       H = Hole (game over, reward = 0)
H  F  F  G       G = Goal (reward = 1.0)
```

- **State Space**: 16 states (positions 0–15 on the 4×4 grid)
- **Action Space**: 4 actions — Left (0), Down (1), Right (2), Up (3)
- **Reward**: `1.0` only when reaching the Goal `G`, otherwise `0.0`
- **Slippery**: `is_slippery=True` — the agent can slide sideways (stochastic transitions)

### State Numbering

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

State 0 = top-left (Start), State 15 = bottom-right (Goal)

---

## ⚙️ Key Concepts & Parameters

### 1. Learning Rate (α — Alpha)

Controls **how much new information overrides old information** in the Q-table.

```python
learning_rate = 0.1  # alpha
```

| α Value | Effect |
|---------|--------|
| `α = 0` | Agent never learns (Q-values never change) |
| `α = 1` | Agent completely overwrites old Q-values per step |
| `α ≈ 0.1` | Stable, gradual learning (used here) |

> **Low α** → Slow but stable learning  
> **High α** → Fast but potentially unstable learning

**In the Bellman Equation:**
```
New_Q = Old_Q + α × (Target - Old_Q)
                    │
                    └── TD Error (Temporal Difference Error)
```

---

### 2. Discount Factor (γ — Gamma)

Controls **how much future rewards matter** vs immediate rewards.

```python
discount_factor = 0.99  # gamma
```

| γ Value | Effect |
|---------|--------|
| `γ = 0` | Agent only cares about immediate rewards (myopic) |
| `γ = 1` | Agent values all future rewards equally as present |
| `γ = 0.99` | Strongly values future rewards (used here) |

> A high γ (like 0.99) is critical for FrozenLake because the reward `+1` only comes at the very end. The agent must learn to plan many steps ahead.

---

### 3. Epsilon-Greedy Exploration Strategy (ε — Epsilon)

Controls the **exploration vs. exploitation trade-off**.

```python
epsilon       = 1.0    # initial exploration rate
max_epsilon   = 1.0    # start fully random
min_epsilon   = 0.01   # minimum exploration floor
epsilon_decay = 0.001  # decay rate
```

| ε Value | Behaviour |
|---------|-----------|
| `ε = 1.0` | 100% random actions (full exploration) |
| `ε = 0.0` | 100% greedy (pure exploitation of learned Q-values) |
| `ε = 0.1` | 10% exploration, 90% exploitation |

### Epsilon Decay Formula

```python
epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
```

This is an **exponential decay** function:
- Episode 0: ε ≈ 1.0 (mostly exploring)
- Episode 1000: ε ≈ 0.37
- Episode 2000: ε ≈ 0.14 (mostly exploiting what it learned)

---

## 📝 Code Walkthrough — Step by Step

### Step 1: Import Libraries

```python
import gymnasium as gym
import numpy as np
import time
import os
```

- **`gymnasium`** — Provides the FrozenLake environment
- **`numpy`** — Used for the Q-table (matrix operations, argmax, zeros)
- **`time`** — Adds delays for visualization (`time.sleep`)
- **`os`** — Used for clearing the terminal display (`os.system('cls'/'clear')`)

---

### Step 2: Create the Environment

```python
env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=True)
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `"FrozenLake-v1"` | env name | The 4×4 grid world |
| `render_mode="ansi"` | text output | Renders environment as ASCII in the terminal |
| `is_slippery=True` | stochastic | Agent may slide in unintended direction (harder problem) |

---

### Step 3: Define Hyperparameters

```python
num_episodes          = 2000   # total training episodes
max_steps_per_episode = 100    # max steps before truncation
learning_rate         = 0.1    # alpha
discount_factor       = 0.99   # gamma
epsilon               = 1.0    # initial epsilon
max_epsilon           = 1.0
min_epsilon           = 0.01
epsilon_decay         = 0.001
```

- `num_episodes = 2000`: The agent plays 2000 complete games
- `max_steps_per_episode = 100`: Prevents infinite looping in a single episode

---

### Step 4: Initialize the Q-Table

```python
state_size  = env.observation_space.n    # = 16
action_size = env.action_space.n         # = 4
Q = np.zeros((state_size, action_size))  # 16 × 4 matrix of zeros
```

The Q-table is a **16 × 4 matrix** — one row per state, one column per action:

```
        Left  Down  Right  Up
State 0  [0.0,  0.0,  0.0,  0.0]
State 1  [0.0,  0.0,  0.0,  0.0]
...
State 15 [0.0,  0.0,  0.0,  0.0]
```

It starts as all zeros and gets updated during training using the Bellman equation.

---

### Step 5: The Training Loop

```python
for episode in range(num_episodes):
    state, _ = env.reset()      # reset to start (state 0)
    done = False
    total_rewards = 0

    for step in range(max_steps_per_episode):
        # ── Epsilon-Greedy Action Selection ──
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()   # EXPLORE: random action
        else:
            action = np.argmax(Q[state, :])      # EXPLOIT: best known action

        # ── Take the action ──
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── Update Q-table (Bellman Equation) ──
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action]
        )

        state = new_state
        total_rewards += reward

        if done:
            break

    # ── Decay Epsilon after each episode ──
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
```

---

### Step 6: Testing the Trained Agent

```python
state, _ = env.reset()
for step in range(max_steps_per_episode):
    action = np.argmax(Q[state, :])   # PURE EXPLOITATION: always use best Q-value
    new_state, reward, terminated, truncated, info = env.step(action)
    state = new_state
    if terminated or truncated:
        print("Test episode ended.")
        break
env.close()
```

During testing, `epsilon = 0` — the agent only follows what it has learned (pure greedy policy).

---

## 📊 How the Q-Table Works

### Before Training (Episode 1)
All Q-values are `0.0` — the agent has no preference for any action.

```
Q-table (initial):
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 ...
 [0. 0. 0. 0.]]
```

### After Training (Episode 2000)
The agent has learned that certain actions from certain states lead closer to the goal:

```
Final Q-table (example):
[[0.52  0.48  0.47  0.46]   ← State 0 (Start): "Left" has highest value
 [0.33  0.21  0.29  0.43]   ← State 1
 ...
 [0.64  0.82  0.75  0.84]   ← State 14 (near Goal): "Down" or "Up" preferred
 [0.   0.   0.   0.  ]]     ← State 15 (Goal): terminal, no future reward
```

- **Hole states** (5, 7, 11, 12) always have `[0. 0. 0. 0.]` because they are terminal with reward 0
- **Goal state** (15) also has `[0. 0. 0. 0.]` as there are no further transitions

---

## 🎲 Epsilon-Greedy Strategy

The **Explore-Exploit Dilemma**:

- **Explore**: Take a random action to discover new paths and rewards
- **Exploit**: Use the current best known action (greedy with respect to Q)

```python
if np.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()   # random (explore)
else:
    action = np.argmax(Q[state, :])      # greedy (exploit)
```

### Epsilon Over Training

```
Episode    Epsilon     Behaviour
─────────────────────────────────
    0      1.0000     100% random exploration
  500      0.6065     60% explore, 40% exploit
 1000      0.3679     37% explore, 63% exploit
 1500      0.2231     22% explore, 78% exploit
 2000      0.1353     14% explore, 86% exploit
```

This gradual shift ensures the agent **explores sufficiently early on** (discovering the reward) and then **exploits what it has learned** as training matures.

---

## 🔁 Training Loop Explained

Each **episode** represents one full game from start to terminal state:

```
Episode Start
    │
    ▼
env.reset() → state = 0 (top-left corner)
    │
    ▼
┌──────────────────────────────────────────────┐
│  For each step:                               │
│  1. Choose action (ε-greedy)                  │
│  2. Execute action → get new_state, reward    │
│  3. Update Q[state, action] via Bellman eq    │
│  4. state = new_state                         │
│  5. If done (hole or goal) → break            │
└──────────────────────────────────────────────┘
    │
    ▼
Decay epsilon: ε = min_ε + (max_ε - min_ε) × e^(-decay × episode)
    │
    ▼
Next episode...
```

### Terminal Conditions
| Condition | `terminated` | `truncated` | Reason |
|-----------|-------------|-------------|--------|
| Falls in hole (H) | `True` | `False` | Game over |
| Reaches goal (G) | `True` | `False` | Success! |
| Exceeds 100 steps | `False` | `True` | Time limit |

---

## 📦 Libraries & Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.x | Core programming language |
| **Gymnasium** | ≥0.26 | RL environment toolkit (successor to OpenAI Gym) |
| **NumPy** | ≥1.21 | Q-table as ndarray; `np.zeros`, `np.argmax`, `np.max`, `np.exp`, `np.random.uniform` |
| **time** | stdlib | `time.sleep(0.05)` to slow rendering for visibility |
| **os** | stdlib | `os.system('cls'/'clear')` to clear terminal between steps |
| **Google Colab** | — | Notebook environment (`.ipynb` format, colab metadata present) |

### Key NumPy Functions Used

| Function | Where Used | Purpose |
|----------|-----------|---------|
| `np.zeros((16,4))` | Q-table init | Creates a 16×4 matrix of all zeros |
| `np.argmax(Q[state, :])` | Action selection | Returns index of highest Q-value for a given state |
| `np.max(Q[new_state, :])` | Bellman update | Gets the maximum Q-value from the next state |
| `np.random.uniform(0, 1)` | ε-greedy | Generates random float between 0 and 1 |
| `np.exp(-decay * episode)` | ε decay | Exponential decay function |

---

## 🚀 Running the Notebook

### Prerequisites

```bash
pip install gymnasium numpy
```

### Option 1: Google Colab (Recommended)
1. Upload `Q_Learning_FrozenLake.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run all cells (`Runtime → Run All`)
3. Watch the agent train live in the output cell!

### Option 2: Local Jupyter

```bash
pip install jupyter gymnasium numpy
jupyter notebook Q_Learning_FrozenLake.ipynb
```

### Option 3: VS Code
1. Open `Q_Learning_FrozenLake.ipynb` in VS Code
2. Select Python kernel
3. Run cells with `Shift + Enter`

> **Note:** The `render_mode="ansi"` renders the grid as colored ASCII text. ANSI colors display correctly in terminals and Colab, but may appear as escape codes in some notebook viewers.

---

## 📈 Results & Output

During training, the following is printed at each step:

```
Episode: 1996/2000 | Step: 19
  (Up)
SFFF
FHFH
F[F]FH        ← agent is at position 9 (highlighted in red)
HFFG

Reward this step: 0.0
Total rewards so far: 0.0
Epsilon: 0.1448

Q-table:
[[0.512  0.468  0.470  0.467]
 [0.327  0.212  0.292  0.416]
 ...
 [0.642  0.820  0.748  0.838]
 [0.     0.     0.     0.   ]]
```

### Final Q-Table (after 2000 episodes)

```
[[0.518  0.479  0.474  0.463]   State  0 – Start
 [0.327  0.213  0.292  0.426]   State  1
 [0.354  0.265  0.254  0.292]   State  2
 [0.027  0.111  0.034  0.310]   State  3
 [0.539  0.354  0.290  0.342]   State  4
 [0.0    0.0    0.0    0.0  ]   State  5 – HOLE
 [0.203  0.174  0.346  0.089]   State  6
 [0.0    0.0    0.0    0.0  ]   State  7 – HOLE
 [0.439  0.422  0.381  0.578]   State  8
 [0.257  0.654  0.387  0.322]   State  9  ← Down is best
 [0.574  0.371  0.415  0.324]   State 10
 [0.0    0.0    0.0    0.0  ]   State 11 – HOLE
 [0.0    0.0    0.0    0.0  ]   State 12 – HOLE
 [0.442  0.357  0.742  0.525]   State 13 ← Right is best
 [0.642  0.820  0.748  0.839]   State 14 ← Down/Up preferred
 [0.0    0.0    0.0    0.0  ]]  State 15 – GOAL (terminal)
```

---

## 💡 Key Takeaways

1. **Q-Learning converges** even in stochastic environments (slippery ice), but needs many episodes
2. **High γ (0.99)** is essential here — rewards only come at the end, so the agent must value future rewards
3. **Epsilon decay** is a proven strategy to balance exploration early and exploitation late
4. **Hole states always have Q = 0** because they are absorbing terminal states with no future reward
5. **State 14 (just above Goal)** has the highest Q-values, confirming the agent learned the optimal path
6. **The slippery nature** means even a trained agent may fail occasionally — this is expected in stochastic MDPs

---

## 📖 References & Further Reading

- [Gymnasium Documentation — FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [Q-Learning — Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
- [Reinforcement Learning: An Introduction — Sutton & Barto (Free PDF)](http://incompleteideas.net/book/the-book-2nd.html)
- [Bellman Equation — Wikipedia](https://en.wikipedia.org/wiki/Bellman_equation)
- [Exploration vs Exploitation Dilemma — Towards Data Science](https://towardsdatascience.com/exploration-vs-exploitation-3de6a9e60c75)

---

## 👤 Author

**Experiment 2** — Reinforcement Learning Lab  
Implementing Q-Learning on FrozenLake-v1 with Epsilon-Greedy Exploration

---

*Made with ❤️ using Python, Gymnasium & NumPy*
