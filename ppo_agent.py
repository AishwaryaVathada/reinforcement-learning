"""
PPO Agent for Sustainable Fishing
=======================================================

"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import os

try:
    from oceanrl import query as ocean_query
    print("✅ oceanrl imported successfully")
except ImportError:
    print("❌ ERROR: oceanrl not found! Install the wheel file.")
    ocean_query = None

# Project specifications
K1, K2, K3, K4 = 0.001, 0.01, 100.0, 100.0
EPISODE_LENGTH_MONTHS = 900
MAX_EFFORT = 1e6

# Initial population ranges (FIXED)
INITIAL_SALMON_MIN = 10_000
INITIAL_SALMON_MAX = 30_000
INITIAL_SHARK_MIN = 400
INITIAL_SHARK_MAX = 600


class SalmonSharkEnv(gym.Env):
    """
    Gymnasium Environment for Sustainable Fishing.
    
    CRITICAL FIX: Months are 1-indexed (1-900) when calling oceanrl.query()
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        
        # Observation: [log1p(salmon), log1p(shark), sin(month), cos(month)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Action: fishing effort [0, MAX_EFFORT]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([MAX_EFFORT], dtype=np.float32)
        )
        
        self.salmon_population = None
        self.shark_population = None
        self.current_month = None
        self.timestep = 0
    
    def _encode_observation(self, salmon: float, shark: float, month: int) -> np.ndarray:
        """Encode state as observation vector"""
        s = math.log1p(max(0.0, salmon))
        k = math.log1p(max(0.0, shark))
        m = month % 12
        sin_m = math.sin(2 * math.pi * m / 12.0)
        cos_m = math.cos(2 * math.pi * m / 12.0)
        return np.array([s, k, sin_m, cos_m], dtype=np.float32)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.salmon_population = float(
            self.rng.integers(INITIAL_SALMON_MIN, INITIAL_SALMON_MAX)
        )
        self.shark_population = float(
            self.rng.integers(INITIAL_SHARK_MIN, INITIAL_SHARK_MAX)
        )
        
        self.current_month = 0
        self.timestep = 0
        
        obs = self._encode_observation(
            self.salmon_population, self.shark_population, self.current_month
        )
        return obs, {}
    
    def step(self, action):
        """
        Execute one timestep.
        
        CRITICAL FIX: oceanrl.query() expects 1-indexed months (1-900)
        """
        effort = float(np.clip(action[0], 0.0, MAX_EFFORT))
        
        # FIXED: Convert to 1-indexed month
        month_for_query = self.current_month + 1
        
        salmon_caught, next_salmon, next_shark = ocean_query(
            self.salmon_population,
            self.shark_population,
            effort,
            month_for_query
        )
        
        reward = K1 * salmon_caught - K2 * effort
        
        self.salmon_population = float(next_salmon)
        self.shark_population = float(next_shark)
        self.current_month += 1
        self.timestep += 1
        
        terminated = self.timestep >= EPISODE_LENGTH_MONTHS
        
        if terminated:
            terminal_bonus = (
                K3 * math.log(max(1e-10, self.salmon_population)) +
                K4 * math.log(max(1e-10, self.shark_population))
            )
            reward += terminal_bonus
        
        obs = self._encode_observation(
            self.salmon_population, self.shark_population, self.current_month
        )
        
        info = {
            "salmon_caught": float(salmon_caught),
            "effort": effort,
            "salmon": self.salmon_population,
            "shark": self.shark_population
        }
        
        return obs, float(reward), terminated, False, info


def make_env(seed: int, rank: int) -> Callable:
    """Create a single monitored environment"""
    def _init():
        env = SalmonSharkEnv(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def make_vec_envs(n_envs: int, seed: int):
    """Create vectorized environments"""
    if n_envs <= 1:
        return DummyVecEnv([make_env(seed, 0)])
    return SubprocVecEnv([make_env(seed, i) for i in range(n_envs)])


class PPOTrainingCallback(BaseCallback):
    """Callback for tracking training progress"""
    
    def __init__(self, save_freq=50_000, save_dir="./models", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_returns = []
        self.episode_lengths = []
        self.last_print = 0
        self.print_freq = 10_000
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_print >= self.print_freq:
            print(f"Training: {self.num_timesteps:>8,} steps - "
                  f"Episodes: {len(self.episode_returns)}")
            self.last_print = self.num_timesteps
        
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    if "episode" in info:
                        ep_return = info["episode"]["r"]
                        ep_length = info["episode"]["l"]
                        
                        self.episode_returns.append(ep_return)
                        self.episode_lengths.append(ep_length)
                        
                        episode_num = len(self.episode_returns)
                        print(f"\nEPISODE {episode_num} - Return: {ep_return:,.2f}, "
                              f"Length: {ep_length:.0f}")
                        
                        if episode_num >= 10:
                            recent_mean = np.mean(self.episode_returns[-10:])
                            print(f"  Avg(10): {recent_mean:,.2f}")
        
        if self.num_timesteps % self.save_freq == 0:
            save_path = f"{self.save_dir}/ppo_checkpoint_{self.num_timesteps}.zip"
            self.model.save(save_path)
            print(f"💾 Checkpoint: {self.num_timesteps} steps")
        
        return True


def train_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    device: str = "auto",
    seed: int = 42
):
    """Train PPO agent"""
    
    print("\n" + "="*70)
    print("TRAINING PPO AGENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Device: {device}\n")
    
    vec_env = make_vec_envs(n_envs, seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )
    
    # PPO is on-policy, so n_steps should be tuned for episode length
    # For 900-step episodes, use 2048 or 4096
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,  # Rollout buffer size per env
        batch_size=64,
        n_epochs=10,
        gamma=1.0,  # No discounting
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh
        ),
        device=device,
        tensorboard_log="./ppo_logs/"
    )
    
    callback = PPOTrainingCallback(save_freq=50_000, save_dir="./models")
    
    print("🚀 Starting training...\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    model.save("./models/ppo_fishing_model.zip")
    vec_env.save("./models/ppo_vecnorm.pkl")
    
    print("\n✅ Training complete!")
    print("💾 Saved: models/ppo_fishing_model.zip and ppo_vecnorm.pkl\n")
    
    return model, vec_env


class PPOFishingAgent:
    """
    PPO Agent for professor's evaluation.
    """
    
    def __init__(self, model_path: str = "./models/ppo_fishing_model.zip"):
        """Initialize with trained model"""
        print(f"Loading PPO model from: {model_path}")
        
        dummy_env = DummyVecEnv([lambda: SalmonSharkEnv()])
        
        vecnorm_path = model_path.replace("_model.zip", "_vecnorm.pkl")
        if os.path.exists(vecnorm_path):
            self.vec_norm = VecNormalize.load(vecnorm_path, dummy_env)
            self.vec_norm.training = False
            self.vec_norm.norm_reward = False
            print(f"✅ Loaded normalization stats")
        else:
            self.vec_norm = None
        
        self.model = PPO.load(model_path)
        print("✅ PPO agent initialized\n")
    
    def _encode_observation(self, salmon: float, shark: float, month: int) -> np.ndarray:
        """Encode state as observation vector"""
        s = math.log1p(max(0.0, salmon))
        k = math.log1p(max(0.0, shark))
        m = month % 12
        sin_m = math.sin(2 * math.pi * m / 12.0)
        cos_m = math.cos(2 * math.pi * m / 12.0)
        return np.array([s, k, sin_m, cos_m], dtype=np.float32)
    
    def act(self, state: Tuple[float, float, int]) -> float:
        """
        Decide fishing effort given current state.
        Matches starter kit format exactly.
        """
        salmon_t, shark_t, month_t = state
        
        obs = self._encode_observation(salmon_t, shark_t, month_t)
        
        if self.vec_norm is not None:
            obs = self.vec_norm.normalize_obs(obs[None, :])[0]
        
        action, _ = self.model.predict(obs, deterministic=True)
        
        effort = float(np.clip(action[0], 0.0, MAX_EFFORT))
        return effort


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO for Sustainable Fishing")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    
    if args.train:
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            device=args.device
        )
    
    if args.eval:
        if ocean_query is None:
            print("❌ Cannot evaluate without oceanrl!")
        else:
            agent = PPOFishingAgent()
            
            salmon_t, shark_t = 20000.0, 500.0
            total_caught = 0.0
            total_effort = 0.0
            
            for month_t in range(1, 901):
                effort = agent.act((salmon_t, shark_t, month_t))
                caught, salmon_t, shark_t = ocean_query(
                    salmon_t, shark_t, effort, month_t
                )
                total_caught += caught
                total_effort += effort
            
            G = (K1 * total_caught - K2 * total_effort +
                 K3 * math.log(max(1e-10, salmon_t)) +
                 K4 * math.log(max(1e-10, shark_t)))
            
            print(f"\nTotal Return: {G:,.2f}")
            print(f"Final Salmon: {salmon_t:,.0f}, Sharks: {shark_t:,.0f}\n")
