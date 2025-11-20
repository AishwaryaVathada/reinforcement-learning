# Sustainable Fishing: Deep Reinforcement Learning for Ecosystem Management

## Project Overview

This repository contains a comprehensive implementation of deep reinforcement learning algorithms designed to solve the sustainable fishing problem - balancing immediate resource extraction with long-term ecosystem preservation. The project implements multiple state-of-the-art RL algorithms (SAC, TD3, TQC, PPO) with proper hyperparameter optimization and evaluation frameworks.

### Problem Statement

The sustainable fishing problem models a simplified ocean ecosystem containing two species:
- **Salmon (Prey)**: Population dynamics affected by reproduction, natural mortality, predation by sharks, and human fishing
- **Sharks (Predators)**: Population dynamics affected by prey abundance and natural factors

Human fishing effort introduces a controllable factor that must be optimized over a 75-year (900-month) horizon to maximize total return while maintaining ecosystem sustainability.

### Ecological Dynamics

Real-world fishing ecosystems exhibit complex predator-prey dynamics, seasonality effects, and long-term equilibrium states:

1. **Lotka-Volterra Dynamics**: The classical predator-prey model describes oscillating populations where prey abundance drives predator growth, and predator abundance controls prey populations.

2. **Seasonal Variations**: Fish reproduction and migration patterns vary significantly by season, affecting both population growth rates and catch efficiency.

3. **Tipping Points**: Overfishing can push ecosystems past critical thresholds, leading to population collapse and difficult recovery.

4. **Economic Trade-offs**: Short-term profit maximization through intensive fishing conflicts with long-term sustainability and ecosystem health.

**Key References**:
- Lotka, A. J. (1925). Elements of Physical Biology. Williams & Wilkins.
- Volterra, V. (1926). Fluctuations in the Abundance of a Species considered Mathematically. Nature, 118(2972), 558-560.
- Schaefer, M. B. (1954). Some Aspects of the Dynamics of Populations Important to the Management of the Commercial Marine Fisheries. Bulletin of the Inter-American Tropical Tuna Commission, 1(2), 27-56.
- Hilborn, R., & Walters, C. J. (1992). Quantitative Fisheries Stock Assessment: Choice, Dynamics and Uncertainty. Chapman and Hall.

---

## Mathematical Formulation

### State Space

The environment state at time t is represented as:
```
s_t = (salmon_t, shark_t, month_t)
```

where:
- `salmon_t`: Number of salmon in the ocean at month t
- `shark_t`: Number of sharks in the ocean at month t  
- `month_t`: Current month (1-900)

### Action Space

The action space is continuous and unbounded:
```
a_t = fishing_effort_t ∈ [0, ∞)
```

where `fishing_effort_t` represents the resources (e.g., boats, nets, labor) dedicated to fishing. Note that effort has diminishing returns - 10x effort does not yield 10x catch.

### Dynamics

The ecosystem dynamics are governed by the oceanrl simulator:
```python
(salmon_caught_t, salmon_{t+1}, shark_{t+1}) = query(salmon_t, shark_t, fishing_effort_t, month_t)
```

The internal dynamics incorporate:
- Salmon reproduction (seasonally varying)
- Natural mortality for both species
- Predation (sharks eating salmon)
- Human fishing (salmon caught)
- Long-term cycles and environmental factors

### Reward Function

The immediate reward at each timestep is:
```
r_t = K1 × salmon_caught_t - K2 × fishing_effort_t
```

where:
- `K1 = 0.001`: Economic value per salmon caught
- `K2 = 0.01`: Cost per unit of fishing effort

### Terminal Reward

Upon episode termination (t = 900), a sustainability bonus is added:
```
r_terminal = K3 × log(salmon_900) + K4 × log(shark_900)
```

where:
- `K3 = 100.0`: Weight for salmon population sustainability
- `K4 = 100.0`: Weight for shark population sustainability

### Total Return

The objective is to maximize the undiscounted total return:
```
G = Σ(t=1 to 900) r_t + r_terminal
```

Note: No temporal discounting is applied (γ = 1.0), emphasizing long-term planning.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Step 1: Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv fishing_env
source fishing_env/bin/activate  # On Windows: fishing_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy<2.0
torch>=2.0.0
gymnasium>=0.28.0
stable-baselines3>=2.0.0
sb3-contrib>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
optuna>=3.0.0
```

### Step 3: Install oceanrl Simulator

The `oceanrl` package is distributed as a wheel file and must be obtained from https://medium.com/mitb-for-all/building-python-wheel-for-third-party-execution-4669b3d64cc9. Once obtained:

```bash
pip install oceanrl-0.1.0-py3-none-any.whl
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import gymnasium; print('Gymnasium:', gymnasium.__version__)"
python -c "from stable_baselines3 import SAC, TD3, PPO; print('Stable-Baselines3: OK')"
python -c "from sb3_contrib import TQC; print('SB3-Contrib: OK')"
python -c "from oceanrl import query; print('oceanrl: OK')"
```

All imports should complete without errors.

---

## Project Structure

```
sustainable-fishing-rl/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── src/                        # Source code
│   ├── sac_agent.py            # Soft Actor-Critic implementation
│   ├── td3_agent.py            # Twin Delayed DDPG implementation
│   ├── tqc_agent.py            # Truncated Quantile Critics implementation
│   ├── ppo_agent.py            # Proximal Policy Optimization implementation
│   └── optuna_tuner.py         # Universal hyperparameter optimization
│
|
├── models/                     # Trained model weights (created after training)
│   ├── sac_fishing_model.zip  
│   ├── sac_vecnorm.pkl
│   ├── td3_fishing_model.zip
│   ├── td3_vecnorm.pkl
│   ├── tqc_fishing_model.zip
│   ├── tqc_vecnorm.pkl
│   ├── ppo_fishing_model.zip
│   └── ppo_vecnorm.pkl
│
|
├── optuna_results/             # Hyperparameter optimization results
│   ├── sac_best_params.json
│   ├── td3_best_params.json
│   ├── tqc_best_params.json
│   └── ppo_best_params.json
│
|
├── logs/                       # Training logs and TensorBoard data
└── results/                    # Evaluation outputs and visualizations
```

---

## Algorithms

This project implements four state-of-the-art deep RL algorithms, each with distinct characteristics:

### 1. Soft Actor-Critic (SAC)

**Type**: Off-policy, Actor-Critic  
**Paper**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018)

**Key Features**:
- Entropy regularization encourages exploration
- Sample-efficient learning from replay buffer
- Automatic temperature tuning
- Stochastic policy (naturally handles uncertainty)

**Best For**: General-purpose continuous control with good exploration

**Implementation**: `src/sac_agent.py`

### 2. Twin Delayed Deep Deterministic Policy Gradient (TD3)

**Type**: Off-policy, Actor-Critic  
**Paper**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018)

**Key Features**:
- Twin Q-networks reduce overestimation bias
- Delayed policy updates improve stability
- Target policy smoothing
- Deterministic policy

**Best For**: Tasks requiring deterministic behavior and stable training

**Implementation**: `src/td3_agent.py`

### 3. Truncated Quantile Critics (TQC)

**Type**: Off-policy, Distributional RL  
**Paper**: Kuznetsov et al., "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics" (2020)

**Key Features**:
- Distributional value estimates (models full return distribution)
- Truncated quantile regression
- Superior handling of uncertainty
- State-of-the-art performance

**Best For**: Tasks with high uncertainty or requiring risk-aware policies

**Implementation**: `src/tqc_agent.py`

### 4. Proximal Policy Optimization (PPO)

**Type**: On-policy, Actor-Critic  
**Paper**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

**Key Features**:
- Clipped surrogate objective prevents large policy updates
- Generalized Advantage Estimation (GAE)
- On-policy learning (uses current experience)
- Very stable training

**Best For**: Tasks requiring stable training and consistent results

**Implementation**: `src/ppo_agent.py`

### Algorithm Comparison

| Algorithm | Sample Efficiency | Training Stability | Exploration | Computational Cost | Recommended Use Case |
|-----------|------------------|-------------------|-------------|-------------------|---------------------|
| **SAC**   | High             | High              | Excellent   | Medium            | General purpose, exploration-heavy tasks |
| **TD3**   | High             | High              | Good        | Low               | Deterministic control, fast training |
| **TQC**   | Very High        | High              | Excellent   | High              | Uncertainty quantification, risk-aware decisions |
| **PPO**   | Medium           | Very High         | Good        | Medium            | Stable training, safety-critical applications |

---

## Usage

### Training an Agent

Each algorithm can be trained independently with command-line arguments:

#### SAC (Recommended)
```bash
python src/sac_agent.py --train --timesteps 1000000 --n-envs 4
```

#### TD3
```bash
python src/td3_agent.py --train --timesteps 1000000 --n-envs 4 --device cuda
```

#### TQC
```bash
python src/tqc_agent.py --train --timesteps 1000000 --n-envs 4
```

#### PPO
```bash
python src/ppo_agent.py --train --timesteps 1000000 --n-envs 8
```

**Training Arguments**:
- `--train`: Enable training mode
- `--timesteps`: Total training timesteps (default: 1,000,000)
- `--n-envs`: Number of parallel environments (default: 4 for off-policy, 8 for PPO)
- `--device`: Compute device - `auto`, `cpu`, or `cuda` (default: auto)
- `--eval`: Run evaluation after training

**Expected Training Time** (1M timesteps):
- CPU (8 cores): 4-8 hours
- GPU (CUDA): 1-3 hours

### Evaluating a Trained Agent

After training, evaluate the agent's performance:

```bash
python src/sac_agent.py --eval
```

This will:
1. Load the trained model from `models/sac_fishing_model.zip`
2. Run a full 900-month episode
3. Report total return and final populations
4. Display evaluation statistics

### Hyperparameter Optimization

Use Optuna for automated hyperparameter tuning:

```bash
# Tune SAC with 50 trials
python src/optuna_tuner.py --algo sac --trials 50 --timesteps 100000

# Tune TD3 with GPU acceleration
python src/optuna_tuner.py --algo td3 --trials 30 --device cuda --timesteps 50000

# Tune TQC
python src/optuna_tuner.py --algo tqc --trials 20 --timesteps 100000

# Tune PPO
python src/optuna_tuner.py --algo ppo --trials 40 --timesteps 100000
```

**Optimization Parameters**:
- `--algo`: Algorithm to optimize (sac/td3/tqc/ppo)
- `--trials`: Number of Optuna trials
- `--timesteps`: Training timesteps per trial
- `--eval-episodes`: Number of evaluation episodes per trial (default: 3)
- `--device`: Compute device

Results are saved to `optuna_results/{algo}_best_params.json`.

---

## Agent Interface

All trained agents implement a standardized interface compatible with the evaluation framework:

```python
from src.sac_agent import SustainableFishingAgent
from oceanrl import query

# Initialize agent with trained weights
agent = SustainableFishingAgent(model_path="./models/sac_fishing_model.zip")

# Run evaluation episode
salmon_t, shark_t = 20000.0, 500.0
total_caught = 0.0
total_effort = 0.0

for month_t in range(1, 901):
    # Agent returns fishing effort for current state
    fishing_effort = agent.act((salmon_t, shark_t, month_t))
    
    # Query ecosystem simulator
    caught, salmon_t, shark_t = query(
        salmon_t, shark_t, fishing_effort, month_t
    )
    
    total_caught += caught
    total_effort += fishing_effort

# Calculate final return
K1, K2, K3, K4 = 0.001, 0.01, 100.0, 100.0
G = (K1 * total_caught - K2 * total_effort + 
     K3 * np.log(salmon_t) + K4 * np.log(shark_t))

print(f"Total Return: {G:.2f}")
print(f"Final Salmon: {salmon_t:.0f}")
print(f"Final Sharks: {shark_t:.0f}")
```

**Key Methods**:
- `agent.act(state)`: Returns fishing effort given `(salmon_t, shark_t, month_t)`
- Returns: `float` representing fishing effort (non-negative)

---

## Implementation Details

### Critical Bug Fixes

This implementation addresses several critical issues identified in preliminary testing:

#### 1. Month Indexing Bug

**Problem**: The oceanrl simulator expects 1-indexed months (1-900), but the environment was internally tracking 0-indexed months (0-899).

**Solution**:
```python
# Environment tracks 0-899 internally
self.current_month = 0  # Internal counter

# Convert to 1-indexed when calling oceanrl
month_for_query = self.current_month + 1  # Pass 1-900 to query()
salmon_caught, next_salmon, next_shark = ocean_query(
    self.salmon_population,
    self.shark_population,
    effort,
    month_for_query  # Correct: 1-900
)
```

#### 2. Population Range Correction

**Problem**: Initial shark population range was incorrectly set.

**Solution**:
```python
# Correct ranges (as specified in project brief)
INITIAL_SALMON_MIN = 10_000
INITIAL_SALMON_MAX = 30_000
INITIAL_SHARK_MIN = 400      # Fixed: was 5_000
INITIAL_SHARK_MAX = 600      # Fixed: was 10_000
```

#### 3. Agent Interface Compatibility

**Problem**: Agent interface must exactly match evaluation script format.

**Solution**:
```python
class SustainableFishingAgent:
    def act(self, state: Tuple[float, float, int]) -> float:
        """
        Standard interface for evaluation.
        Args: state = (salmon_t, shark_t, month_t)
        Returns: fishing_effort (float)
        """
        salmon_t, shark_t, month_t = state
        # ... encode observation ...
        action, _ = self.model.predict(obs, deterministic=True)
        return float(action[0])
```

### Observation Encoding

Raw state is transformed into a rich observation vector:

```python
observation = [
    log1p(salmon_t),           # Log-normalized salmon population
    log1p(shark_t),            # Log-normalized shark population  
    sin(2π × (month % 12) / 12), # Seasonal encoding (sine)
    cos(2π × (month % 12) / 12)  # Seasonal encoding (cosine)
]
```

This encoding:
- Normalizes population scales (log transformation)
- Captures seasonality cyclically (sin/cos prevents discontinuity at month 12→1)
- Provides 4-dimensional continuous observation space

### Network Architecture

All algorithms use Multi-Layer Perceptron (MLP) networks with:
- **Hidden layers**: 2-4 layers with 128-512 neurons per layer
- **Activation**: ReLU (SAC, TD3, TQC) or Tanh (PPO)
- **Separate networks**: Actor and Critic networks trained jointly

Example SAC architecture (from Optuna optimization):
```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256, 256],  # Actor network
        qf=[256, 256, 256]   # Critic network
    ),
    activation_fn=torch.nn.ReLU
)
```

### Training Optimizations

1. **Vectorized Environments**: Multiple parallel environments for faster data collection
2. **Experience Replay**: Off-policy algorithms use replay buffers (capacity: 500K-1M)
3. **Normalization**: Observation normalization with running statistics
4. **Gradient Clipping**: Prevents training instability from large gradients
5. **Checkpointing**: Periodic model saving every 50,000 steps

---

## Expected Performance

### Training Progress

After 1,000,000 training timesteps, expect:

| Algorithm | Avg. Episode Return | Final Salmon | Final Sharks | Training Time (GPU) |
|-----------|-------------------|--------------|-------------|-------------------|
| **SAC**   | 1,600 - 2,000     | 15K - 25K    | 400 - 600   | 1-2 hours        |
| **TD3**   | 1,500 - 1,900     | 15K - 25K    | 400 - 600   | 1-2 hours        |
| **TQC**   | 1,700 - 2,100     | 15K - 25K    | 400 - 600   | 2-3 hours        |
| **PPO**   | 1,400 - 1,800     | 15K - 25K    | 400 - 600   | 2-3 hours        |

### Performance Metrics

**Sustainability Indicators**:
- Final salmon population > 10,000 (healthy population)
- Final shark population > 300 (predator sustainability)
- Total catch: 50,000 - 150,000 salmon over 75 years

**Economic Indicators**:
- Positive total return (profitability)
- Balanced effort allocation across time
- Seasonal catch variation reflecting reproduction cycles

### Learning Curves

Typical learning progression:
- **Episodes 1-10**: Exploration, highly variable returns (often negative)
- **Episodes 10-50**: Policy improvement, increasing returns
- **Episodes 50-100**: Convergence, stable high returns
- **Episodes 100+**: Fine-tuning, near-optimal performance

---

## Troubleshooting

### NumPy Version Conflicts

**Error**: `AttributeError: module 'numpy' has no attribute 'X'`

**Solution**:
```bash
pip uninstall numpy -y
pip install "numpy<2.0"
pip install --force-reinstall matplotlib
```

### oceanrl Not Found

**Error**: `ImportError: No module named 'oceanrl'`

**Solution**:
```bash
# Obtain the wheel file from your instructor, then:
pip install oceanrl-0.1.0-py3-none-any.whl
```

### TQC Import Error

**Error**: `ImportError: cannot import name 'TQC'`

**Solution**:
```bash
pip install sb3-contrib
```

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in code
2. Reduce number of parallel environments: `--n-envs 1`
3. Use CPU instead: `--device cpu`
4. Use smaller network: Modify `layer_size` in code

### Training Too Slow

**Solutions**:
1. Enable GPU: `--device cuda`
2. Increase parallel environments: `--n-envs 8`
3. Use faster algorithm: TD3 or SAC instead of TQC/PPO
4. Reduce logging frequency

---

## Algorithm Selection Guide

### Choose SAC if:
- You want the best all-around performance
- Your task benefits from exploration
- You need a stochastic policy
- You want a proven, reliable algorithm

**Recommendation**: Start here for most applications

### Choose TD3 if:
- You need deterministic behavior
- Training speed is critical
- You have limited compute resources
- You prefer simpler algorithms

### Choose TQC if:
- You want state-of-the-art performance
- Uncertainty quantification is important
- You need risk-aware decision making
- You have sufficient compute resources

### Choose PPO if:
- Training stability is the top priority
- You're new to RL (most forgiving)
- You need consistent, reproducible results
- Safety constraints matter

---

## References

### Academic Papers

1. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. International Conference on Machine Learning (ICML).

2. Fujimoto, S., Van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. International Conference on Machine Learning (ICML).

3. Kuznetsov, A., Shvechikov, P., Grishin, A., & Vetrov, D. (2020). Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics. International Conference on Machine Learning (ICML).

4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.

### Documentation

- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- SB3-Contrib: https://sb3-contrib.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- Optuna: https://optuna.readthedocs.io/

### Ecological References

- Lotka, A. J. (1925). Elements of Physical Biology
- Volterra, V. (1926). Fluctuations in the Abundance of a Species considered Mathematically
- Schaefer, M. B. (1954). Some Aspects of the Dynamics of Populations Important to the Management of the Commercial Marine Fisheries
- Hilborn, R., & Walters, C. J. (1992). Quantitative Fisheries Stock Assessment

---

## License

This project is submitted as coursework for AY25/26 T4. All rights reserved by the project authors.

---

## Authors

AY25/26 Team  
Date: November 2025

---

## Acknowledgments

- Course instructors for providing the oceanrl simulator and project specifications
