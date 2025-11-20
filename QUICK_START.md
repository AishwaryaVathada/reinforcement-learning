# Quick Start Guide

This guide helps you get the project running in under 5 minutes.

---

## Installation (2 minutes)

```bash
# 1. Create virtual environment
python -m venv fishing_env
source fishing_env/bin/activate  # Windows: fishing_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install oceanrl (get wheel file from instructor)
pip install oceanrl-0.1.0-py3-none-any.whl

# 4. Verify installation
python -c "from oceanrl import query; print('oceanrl: OK')"
```

---

## Evaluate Pre-Trained Model (1 minute)

If trained models are included in submission:

```bash
# Evaluate SAC agent
python src/sac_agent.py --eval

# Or evaluate any other algorithm
python src/td3_agent.py --eval
python src/tqc_agent.py --eval
python src/ppo_agent.py --eval
```

Expected output:
```
EVALUATION RESULTS
======================================================================
  Total Return: 1,847.23
  Total Caught: 87,543
  Total Effort: 234,567.89
  Final Salmon: 18,432
  Final Sharks: 487
======================================================================
```

---

## Train New Model (30-60 minutes)

```bash
# Quick training (100K steps, ~5-10 minutes)
python src/sac_agent.py --train --timesteps 100000

# Full training (1M steps, ~30-60 minutes)
python src/sac_agent.py --train --timesteps 1000000

# With GPU acceleration (faster)
python src/sac_agent.py --train --timesteps 1000000 --device cuda
```

---

## Test Agent Interface

Verify the agent works with evaluation format:

```python
# test_agent.py
from src.sac_agent import SustainableFishingAgent
from oceanrl import query

# Load trained agent
agent = SustainableFishingAgent(model_path="./models/sac_fishing_model.zip")

# Test single step
effort = agent.act((20000, 500, 1))
print(f"Test successful! Fishing effort: {effort:.2f}")

# Run short episode (10 months)
salmon, shark = 20000.0, 500.0
for month in range(1, 11):
    effort = agent.act((salmon, shark, month))
    caught, salmon, shark = query(salmon, shark, effort, month)
    print(f"Month {month}: Caught {caught:.0f}, Salmon: {salmon:.0f}, Sharks: {shark:.0f}")
```

Run it:
```bash
python test_agent.py
```

---

## Project Structure Overview

```
project/
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Python dependencies
├── src/
│   ├── sac_agent.py            # Main algorithm (recommended)
│   ├── td3_agent.py            # Alternative algorithm
│   ├── tqc_agent.py            # Alternative algorithm
│   ├── ppo_agent.py            # Alternative algorithm
│   └── optuna_tuner.py         # Hyperparameter optimization
└── models/
    ├── sac_fishing_model.zip   # Trained weights
    └── sac_vecnorm.pkl         # Normalization stats
```

---

## Common Issues

### Issue 1: oceanrl not found
```bash
# Solution: Install the wheel file
pip install oceanrl-0.1.0-py3-none-any.whl
```

### Issue 2: NumPy version error
```bash
# Solution: Downgrade NumPy
pip uninstall numpy -y
pip install "numpy<2.0"
```

### Issue 3: Model file not found
```bash
# Solution: Train a model first
python src/sac_agent.py --train --timesteps 100000
```

### Issue 4: CUDA out of memory
```bash
# Solution: Use CPU or reduce batch size
python src/sac_agent.py --train --device cpu
```

---

## Expected Results

After training 1M steps:

| Metric | Value |
|--------|-------|
| Episode Return | 1,600 - 2,000 |
| Final Salmon | 15,000 - 25,000 |
| Final Sharks | 400 - 600 |
| Total Catch | 50,000 - 150,000 |

---

## Need Help?

1. Check README.md for detailed documentation
2. Review CODE_REVIEW_AND_GUIDE.md for troubleshooting
3. Verify all dependencies are installed correctly

---

## Minimal Working Example

```python
# minimal_test.py - Complete working example
import numpy as np
from src.sac_agent import SustainableFishingAgent
from oceanrl import query

# Initialize
agent = SustainableFishingAgent()
K1, K2, K3, K4 = 0.001, 0.01, 100.0, 100.0

# Run episode
salmon_t, shark_t = 20000.0, 500.0
total_caught, total_effort = 0.0, 0.0

for month_t in range(1, 901):
    effort = agent.act((salmon_t, shark_t, month_t))
    caught, salmon_t, shark_t = query(salmon_t, shark_t, effort, month_t)
    total_caught += caught
    total_effort += effort

# Calculate return
G = (K1 * total_caught - K2 * total_effort + 
     K3 * np.log(salmon_t) + K4 * np.log(shark_t))

print(f"Total Return: {G:.2f}")
print(f"Final Salmon: {salmon_t:.0f}")
print(f"Final Sharks: {shark_t:.0f}")
```

---

That's it! You're ready to go.
