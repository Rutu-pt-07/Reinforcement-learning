# 🤖 Reinforcement Learning Experiments

A collection of Reinforcement Learning experiments implemented in Python using [OpenAI Gymnasium](https://gymnasium.farama.org/). Each experiment lives in its own folder with a dedicated notebook and detailed README.

---

## 📁 Experiments

| # | Folder | Algorithm | Environment | Status |
|---|--------|-----------|-------------|--------|
| 1 | [`FrozenLake-QLearning/`](./FrozenLake-QLearning/) | Q-Learning | FrozenLake-v1 | ✅ Complete |
| 2 | _Coming soon_ | — | — | 🔜 |
| 3 | _Coming soon_ | — | — | 🔜 |

---

## 🧪 Current Experiments

### 1. 🧊 [FrozenLake — Q-Learning](./FrozenLake-QLearning/)

> Train an agent to navigate a slippery frozen lake from Start → Goal while avoiding holes, using the Q-Learning algorithm with Epsilon-Greedy exploration.

- **Algorithm:** Q-Learning (Bellman Equation)
- **Environment:** `FrozenLake-v1` (4×4 grid, slippery)
- **Key Concepts:** Q-table, Learning Rate (α), Discount Factor (γ), Epsilon Decay (ε)
- **Notebook:** [`Q_Learning_FrozenLake.ipynb`](./FrozenLake-QLearning/Q_Learning_FrozenLake.ipynb)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Core language |
| **Gymnasium** | RL environment suite |
| **NumPy** | Q-table and numerical ops |
| **Jupyter / Google Colab** | Notebook execution |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Rutu-pt-07/Reinforcement-learning.git
cd Reinforcement-learning

# Install dependencies
pip install gymnasium numpy jupyter

# Open any experiment notebook
jupyter notebook FrozenLake-QLearning/Q_Learning_FrozenLake.ipynb
```

---

## 📖 References

- [Reinforcement Learning: An Introduction — Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- [Q-Learning — Wikipedia](https://en.wikipedia.org/wiki/Q-learning)

---

*More experiments coming soon — stay tuned! ⭐*
