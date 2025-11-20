"""
SAC Agent for Sustainable Fishing
====================================================

"""

import os
import sys
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import pandas as pd
import time

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

# Verify oceanrl is installed
try:
    from oceanrl import query
    print("✅ oceanrl imported successfully")
except ImportError:
    print("❌ ERROR: oceanrl not found!")
    print("Please install: pip install oceanrl-0.1.0-py3-none-any.whl")
    sys.exit(1)

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# HYPERPARAMETERS (from Optuna optimization)
# ============================================================================

BEST_PARAMS = {
    "learning_rate": 0.0007670855756381101,
    "batch_size": 128,
    "buffer_size": 500000,
    "tau": 0.002554208382613289,
    "ent_coef": 0.036431589706522764,
    "n_layers": 3,
    "layer_size": 256,
    "gradient_steps": 100
}

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration matching project specifications"""
    
    # Reward coefficients (FIXED by professor - DO NOT MODIFY)
    K1 = 0.001  # Reward per salmon caught
    K2 = 0.01   # Cost per unit effort
    K3 = 100.0  # Terminal bonus for salmon
    K4 = 100.0  # Terminal bonus for sharks
    
    # Environment
    EPISODE_LENGTH_MONTHS = 900  # 75 years
    MAX_FISHING_EFFORT = 1e6
    
    # Initial population ranges (FIXED: sharks were wrong before)
    INITIAL_SALMON_MIN = 10_000
    INITIAL_SALMON_MAX = 30_000
    INITIAL_SHARK_MIN = 400
    INITIAL_SHARK_MAX = 600
    
    # Training
    TOTAL_TRAINING_STEPS = 1_000_000
    GAMMA = 0.999
    
    # Reward shaping
    INTERMEDIATE_BONUS_INTERVAL = 100
    INTERMEDIATE_BONUS_SCALE = 0.1
    DANGER_THRESHOLD_SALMON = 3000
    DANGER_THRESHOLD_SHARK = 150
    DANGER_PENALTY = 100.0
    FISHING_CONSISTENCY_BONUS = 0.5
    MIN_SUSTAINABLE_EFFORT = 100
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_FREQ = 100_000

# ============================================================================
# ENVIRONMENT
# ============================================================================

class SustainableFishingEnv(gym.Env):
    """
    Gymnasium Environment for Sustainable Fishing.
    
    CRITICAL FIX: Months are now 1-indexed (1-900) when calling oceanrl.query()
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, seed: int = 0, normalize: bool = True, enable_shaping: bool = True):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.normalize = normalize
        self.enable_shaping = enable_shaping
        self.query_func = query
        
        # State: [salmon_norm, shark_norm, sin_month, cos_month, ratio, progress]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 1.0, 1.0, 1000.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: continuous fishing effort
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([Config.MAX_FISHING_EFFORT], dtype=np.float32),
            dtype=np.float32
        )
        
        # Normalization constants
        self.salmon_mean = 200_000.0
        self.salmon_std = 100_000.0
        self.shark_mean = 600.0
        self.shark_std = 400.0
        
        # State
        self.salmon_population = None
        self.shark_population = None
        self.current_month = None  # This tracks 0-899 internally
        self.episode_history: List[Dict] = []
        self.cumulative_catch = 0.0
        self.cumulative_effort = 0.0
    
    def _encode_month_cyclic(self, month: int) -> Tuple[float, float]:
        """Encode month as sin/cos"""
        angle = 2 * math.pi * (month % 12) / 12.0
        return math.sin(angle), math.cos(angle)
    
    def _normalize_population(self, salmon: float, shark: float) -> Tuple[float, float]:
        """Normalize populations"""
        if self.normalize:
            salmon_norm = (salmon - self.salmon_mean) / self.salmon_std
            shark_norm = (shark - self.shark_mean) / self.shark_std
            return salmon_norm, shark_norm
        return salmon, shark
    
    def reset(self, *, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.salmon_population = float(
            self.rng.integers(Config.INITIAL_SALMON_MIN, Config.INITIAL_SALMON_MAX)
        )
        self.shark_population = float(
            self.rng.integers(Config.INITIAL_SHARK_MIN, Config.INITIAL_SHARK_MAX)
        )
        
        self.current_month = 0  # Internal: 0-899
        self.episode_history = []
        self.cumulative_catch = 0.0
        self.cumulative_effort = 0.0
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        salmon_norm, shark_norm = self._normalize_population(
            self.salmon_population, self.shark_population
        )
        sin_month, cos_month = self._encode_month_cyclic(self.current_month)
        ratio = self.salmon_population / (self.shark_population + 1.0)
        progress = self.current_month / Config.EPISODE_LENGTH_MONTHS
        
        return np.array([
            salmon_norm,
            shark_norm,
            sin_month,
            cos_month,
            min(ratio / 100.0, 1000.0),
            progress
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Return state info"""
        return {
            "salmon": self.salmon_population,
            "shark": self.shark_population,
            "month": self.current_month,
        }
    
    def step(self, action):
        """
        Execute one timestep.
        
        CRITICAL FIX: oceanrl.query() expects 1-indexed months (1-900)
        We track 0-899 internally, but pass (current_month + 1) to query()
        """
        fishing_effort = float(np.clip(action[0], 0, Config.MAX_FISHING_EFFORT))
        
        # FIXED: Convert 0-indexed month to 1-indexed for oceanrl
        month_for_query = self.current_month + 1  # Convert to 1-900
        
        # Query ecosystem dynamics
        salmon_caught, next_salmon, next_shark = self.query_func(
            self.salmon_population,
            self.shark_population,
            fishing_effort,
            month_for_query  # FIXED: Now passes 1-900 instead of 0-899
        )
        
        # Base reward
        immediate_reward = (
            Config.K1 * salmon_caught -
            Config.K2 * fishing_effort
        )
        
        # Reward shaping
        if self.enable_shaping and fishing_effort >= Config.MIN_SUSTAINABLE_EFFORT:
            immediate_reward += Config.FISHING_CONSISTENCY_BONUS
        
        if self.enable_shaping and (self.current_month % Config.INTERMEDIATE_BONUS_INTERVAL == 0):
            sustainability = (
                Config.INTERMEDIATE_BONUS_SCALE * math.log(max(next_salmon, 1.0)) +
                Config.INTERMEDIATE_BONUS_SCALE * math.log(max(next_shark, 1.0))
            )
            immediate_reward += sustainability
        
        if self.enable_shaping:
            if next_salmon < Config.DANGER_THRESHOLD_SALMON:
                immediate_reward -= Config.DANGER_PENALTY
            if next_shark < Config.DANGER_THRESHOLD_SHARK:
                immediate_reward -= Config.DANGER_PENALTY
        
        # Update state
        self.salmon_population = float(next_salmon)
        self.shark_population = float(next_shark)
        self.current_month += 1
        self.cumulative_catch += salmon_caught
        self.cumulative_effort += fishing_effort
        
        # Check termination
        terminated = self.current_month >= Config.EPISODE_LENGTH_MONTHS
        
        # Terminal bonus
        if terminated:
            terminal_bonus = (
                Config.K3 * math.log(max(self.salmon_population, 1.0)) +
                Config.K4 * math.log(max(self.shark_population, 1.0))
            )
            immediate_reward += terminal_bonus
        
        # Track history
        self.episode_history.append({
            'month': self.current_month - 1,  # Store 0-indexed for consistency
            'salmon': self.salmon_population,
            'shark': self.shark_population,
            'effort': fishing_effort,
            'caught': salmon_caught,
            'reward': immediate_reward
        })
        
        return self._get_observation(), float(immediate_reward), terminated, False, self._get_info()

# ============================================================================
# TRAINING CALLBACK
# ============================================================================

class FishingMonitorCallback(BaseCallback):
    """Callback for tracking training"""
    
    def __init__(self, verbose=0, save_freq=10_000, save_dir="./models"):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_salmon_final = []
        self.episode_shark_final = []
        self.timesteps = []
        
        self.last_print_step = 0
        self.print_freq = 5000
    
    def _on_step(self) -> bool:
        # Progress updates
        if self.num_timesteps - self.last_print_step >= self.print_freq:
            print(f"Training: {self.num_timesteps:>8,} / 1,000,000 steps "
                  f"({100*self.num_timesteps/1_000_000:.1f}%) - "
                  f"Episodes: {len(self.episode_returns)}")
            self.last_print_step = self.num_timesteps
        
        # Episode completion
        if self.locals.get("dones")[0]:
            info = self.locals["infos"][0]
            
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                self.episode_returns.append(ep_return)
                self.episode_lengths.append(ep_length)
                self.timesteps.append(self.num_timesteps)
                
                final_salmon = info.get("salmon", 0)
                final_shark = info.get("shark", 0)
                
                if final_salmon > 0:
                    self.episode_salmon_final.append(final_salmon)
                if final_shark > 0:
                    self.episode_shark_final.append(final_shark)
                
                episode_num = len(self.episode_returns)
                print(f"\n{'='*70}")
                print(f"EPISODE {episode_num} (Step: {self.num_timesteps:,})")
                print(f"{'='*70}")
                print(f"  Return: {ep_return:>15,.2f}")
                print(f"  Length: {ep_length:>15,.0f} months")
                print(f"  Salmon: {final_salmon:>15,.0f}")
                print(f"  Sharks: {final_shark:>15,.0f}")
                
                if hasattr(self.training_env.envs[0], 'cumulative_catch'):
                    env = self.training_env.envs[0]
                    print(f"  Caught: {env.cumulative_catch:>15,.0f}")
                    print(f"  Effort: {env.cumulative_effort:>15,.2f}")
                
                if episode_num >= 10:
                    recent_mean = np.mean(self.episode_returns[-10:])
                    print(f"  Avg(10): {recent_mean:>14,.2f}")
                
                print(f"{'='*70}\n")
        
        # Periodic save
        if self.num_timesteps % self.save_freq == 0:
            save_path = f"{self.save_dir}/checkpoint_{self.num_timesteps}.zip"
            self.model.save(save_path)
            print(f"💾 Checkpoint saved: {self.num_timesteps} steps")
        
        return True

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_agent(
    total_timesteps: int = Config.TOTAL_TRAINING_STEPS,
    save_dir: str = "./models",
    log_dir: str = "./logs"
) -> Tuple[SAC, FishingMonitorCallback]:
    """Train SAC agent"""
    
    print("\n" + "="*70)
    print("TRAINING WITH FIXED IMPLEMENTATION")
    print("="*70)
    print("\nBest Hyperparameters:")
    for key, value in BEST_PARAMS.items():
        print(f"  {key:20s}: {value}")
    print()
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(f"{save_dir}/best_params.json", 'w') as f:
        json.dump(BEST_PARAMS, f, indent=2)
    
    def make_env(seed=0):
        def _init():
            env = SustainableFishingEnv(seed=seed, normalize=True, enable_shaping=True)
            return Monitor(env, log_dir)
        return _init
    
    env = DummyVecEnv([make_env(SEED)])
    
    net_arch = [BEST_PARAMS["layer_size"]] * BEST_PARAMS["n_layers"]
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.ReLU,
    )
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=BEST_PARAMS["learning_rate"],
        batch_size=BEST_PARAMS["batch_size"],
        buffer_size=BEST_PARAMS["buffer_size"],
        gamma=Config.GAMMA,
        tau=BEST_PARAMS["tau"],
        ent_coef=BEST_PARAMS["ent_coef"],
        train_freq=1,
        gradient_steps=1,
        target_entropy="auto",
        verbose=1,
        tensorboard_log=f"{log_dir}/sac_fishing",
        device="auto"
    )
    
    callback = FishingMonitorCallback(
        verbose=1,
        save_freq=Config.SAVE_FREQ,
        save_dir=save_dir
    )
    
    print(f"\n🚀 Training for {total_timesteps:,} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed/60:.2f} minutes")
    
    # Save final model with consistent pickle protocol
    final_path = f"{save_dir}/sac_fishing_model.zip"
    model.save(final_path)
    print(f"💾 Final model saved: {final_path}")
    
    return model, callback

# ============================================================================
# SUBMISSION AGENT CLASS (matches starter kit format)
# ============================================================================

class SustainableFishingAgent:
    """
    Agent class for professor's evaluation script.
    
    CRITICAL: This matches the starter kit format exactly.
    """
    
    def __init__(self, model_path: str = "./models/sac_fishing_model.zip"):
        """Initialize with trained model"""
        self.model = SAC.load(model_path)
        self.salmon_mean = 200_000.0
        self.salmon_std = 100_000.0
        self.shark_mean = 600.0
        self.shark_std = 400.0
        self.max_episode_length = 900
    
    def _normalize_population(self, salmon: float, shark: float) -> Tuple[float, float]:
        """Normalize populations"""
        salmon_norm = (salmon - self.salmon_mean) / self.salmon_std
        shark_norm = (shark - self.shark_mean) / self.shark_std
        return salmon_norm, shark_norm
    
    def _encode_month_cyclic(self, month: int) -> Tuple[float, float]:
        """Encode month cyclically"""
        angle = 2 * math.pi * (month % 12) / 12.0
        return math.sin(angle), math.cos(angle)
    
    def _construct_observation(self, salmon: float, shark: float, month: int) -> np.ndarray:
        """Construct observation vector"""
        salmon_norm, shark_norm = self._normalize_population(salmon, shark)
        sin_month, cos_month = self._encode_month_cyclic(month)
        ratio = salmon / (shark + 1.0)
        ratio_norm = min(ratio / 100.0, 1000.0)
        progress = month / self.max_episode_length
        
        return np.array([
            salmon_norm, shark_norm, sin_month, cos_month, ratio_norm, progress
        ], dtype=np.float32)
    
    def act(self, state: Tuple[float, float, int]) -> float:
        """
        CRITICAL: This signature must match starter kit exactly.
        
        The professor's script will call:
            agent = SustainableFishingAgent(model_path="...")
            fishing_effort = agent.act((salmon_t, shark_t, month_t))
        """
        salmon_t, shark_t, month_t = state
        obs = self._construct_observation(salmon_t, shark_t, month_t)
        action, _ = self.model.predict(obs, deterministic=True)
        return float(max(0.0, action[0]))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🐟 Sustainable Fishing RL - FIXED VERSION\n")
    
    # Train
    model, callback = train_agent(
        total_timesteps=Config.TOTAL_TRAINING_STEPS,
        save_dir="./models",
        log_dir="./logs"
    )
    
    print("\n✅ Training complete!")
    print("📁 Files ready for submission:")
    print("   - ./models/sac_fishing_model.zip (trained weights)")
    print("   - ./submission_agent.py (agent class)")
    print("   - ./requirements.txt (dependencies)\n")
