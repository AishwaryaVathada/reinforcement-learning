"""
Universal Optuna Hyperparameter Tuning for RL Agents
=====================================================

Supports: SAC, TD3, TQC, PPO
Automatically tunes hyperparameters for the sustainable fishing problem.

Usage:
    python optuna_tuner.py --algo sac --trials 50 --timesteps 100000
    python optuna_tuner.py --algo td3 --trials 30 --timesteps 50000
    python optuna_tuner.py --algo tqc --trials 20
    python optuna_tuner.py --algo ppo --trials 40

"""

import argparse
import json
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Callable, Dict, Any
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch
import os

try:
    from sb3_contrib import TQC
    TQC_AVAILABLE = True
except ImportError:
    TQC_AVAILABLE = False
    print("⚠️  sb3-contrib not installed, TQC unavailable")

try:
    from oceanrl import query as ocean_query
    print("✅ oceanrl imported")
except ImportError:
    print("❌ oceanrl not found!")
    ocean_query = None

# Constants
K1, K2, K3, K4 = 0.001, 0.01, 100.0, 100.0
EPISODE_LENGTH_MONTHS = 900
MAX_EFFORT = 1e6
INITIAL_SALMON_MIN = 10_000
INITIAL_SALMON_MAX = 30_000
INITIAL_SHARK_MIN = 400
INITIAL_SHARK_MAX = 600


class SalmonSharkEnv(gym.Env):
    """Environment for hyperparameter tuning"""
    
    metadata = {"render_modes": []}
    
    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([MAX_EFFORT], dtype=np.float32)
        )
        
        self.salmon_population = None
        self.shark_population = None
        self.current_month = None
        self.timestep = 0
    
    def _encode_observation(self, salmon: float, shark: float, month: int) -> np.ndarray:
        s = math.log1p(max(0.0, salmon))
        k = math.log1p(max(0.0, shark))
        m = month % 12
        sin_m = math.sin(2 * math.pi * m / 12.0)
        cos_m = math.cos(2 * math.pi * m / 12.0)
        return np.array([s, k, sin_m, cos_m], dtype=np.float32)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
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
        effort = float(np.clip(action[0], 0.0, MAX_EFFORT))
        month_for_query = self.current_month + 1  # FIXED: 1-indexed
        
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
        
        info = {"salmon_caught": float(salmon_caught), "effort": effort}
        
        return obs, float(reward), terminated, False, info


def make_env(seed: int) -> Callable:
    def _init():
        env = SalmonSharkEnv(seed=seed)
        env = Monitor(env)
        return env
    return _init


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample SAC hyperparameters"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 300_000, 500_000, 1_000_000]),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05, 0.1]),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "layer_size": trial.suggest_categorical("layer_size", [128, 256, 512]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 50, 100])
    }


def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample TD3 hyperparameters"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [300_000, 500_000, 1_000_000]),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "policy_delay": trial.suggest_int("policy_delay", 1, 3),
        "noise_sigma": trial.suggest_float("noise_sigma", 0.1, 0.5),
        "n_layers": trial.suggest_int("n_layers", 2, 3),
        "layer_size": trial.suggest_categorical("layer_size", [256, 400, 512])
    }


def sample_tqc_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample TQC hyperparameters"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [300_000, 500_000, 1_000_000]),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "n_critics": trial.suggest_int("n_critics", 2, 5),
        "n_quantiles": trial.suggest_categorical("n_quantiles", [25, 50, 100]),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "layer_size": trial.suggest_categorical("layer_size", [256, 512])
    }


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample PPO hyperparameters"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_epochs": trial.suggest_int("n_epochs", 5, 20),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "n_layers": trial.suggest_int("n_layers", 2, 3),
        "layer_size": trial.suggest_categorical("layer_size", [128, 256, 512])
    }


def create_model(algo: str, env, params: Dict[str, Any], seed: int, device: str):
    """Create model with sampled hyperparameters"""
    
    # Common parameters
    common_params = {
        "env": env,
        "verbose": 0,
        "seed": seed,
        "device": device,
        "gamma": 1.0  # No discounting per project spec
    }
    
    if algo == "sac":
        net_arch = [params["layer_size"]] * params["n_layers"]
        model = SAC(
            policy="MlpPolicy",
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            buffer_size=params["buffer_size"],
            tau=params["tau"],
            ent_coef=params["ent_coef"],
            train_freq=1,
            gradient_steps=params["gradient_steps"],
            policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)),
            **common_params
        )
    
    elif algo == "td3":
        from stable_baselines3.common.noise import NormalActionNoise
        net_arch = [params["layer_size"]] * params["n_layers"]
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=params["noise_sigma"] * np.ones(n_actions)
        )
        model = TD3(
            policy="MlpPolicy",
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            buffer_size=params["buffer_size"],
            tau=params["tau"],
            policy_delay=params["policy_delay"],
            action_noise=action_noise,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=net_arch),
            **common_params
        )
    
    elif algo == "tqc":
        if not TQC_AVAILABLE:
            raise ValueError("TQC requires sb3-contrib: pip install sb3-contrib")
        net_arch = [params["layer_size"]] * params["n_layers"]
        model = TQC(
            policy="MlpPolicy",
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            buffer_size=params["buffer_size"],
            tau=params["tau"],
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=net_arch,
                n_critics=params["n_critics"],
                n_quantiles=params["n_quantiles"]
            ),
            **common_params
        )
    
    elif algo == "ppo":
        net_arch = [params["layer_size"]] * params["n_layers"]
        model = PPO(
            policy="MlpPolicy",
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)),
            **common_params
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return model


def objective(
    trial: optuna.Trial,
    algo: str,
    n_timesteps: int,
    n_eval_episodes: int,
    seed: int,
    device: str
) -> float:
    """Optuna objective function"""
    
    # Sample hyperparameters
    if algo == "sac":
        params = sample_sac_params(trial)
    elif algo == "td3":
        params = sample_td3_params(trial)
    elif algo == "tqc":
        params = sample_tqc_params(trial)
    elif algo == "ppo":
        params = sample_ppo_params(trial)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Create environment
    env = DummyVecEnv([make_env(seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create model
    model = create_model(algo, env, params, seed, device)
    
    # Train
    try:
        model.learn(total_timesteps=n_timesteps, progress_bar=False)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -np.inf
    
    # Evaluate
    eval_env = DummyVecEnv([make_env(seed + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.obs_rms = env.obs_rms  # Copy normalization stats
    eval_env.training = False
    eval_env.norm_reward = False
    
    episode_returns = []
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_return = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_return += reward[0]
        
        episode_returns.append(episode_return)
    
    mean_return = float(np.mean(episode_returns))
    
    # Report intermediate value for pruning
    trial.report(mean_return, n_timesteps)
    
    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    print(f"Trial {trial.number}: Mean Return = {mean_return:.2f}")
    
    return mean_return


def run_optimization(
    algo: str,
    n_trials: int = 50,
    n_timesteps: int = 100_000,
    n_eval_episodes: int = 3,
    seed: int = 42,
    device: str = "auto",
    study_name: Optional[str] = None
):
    """Run Optuna hyperparameter optimization"""
    
    if ocean_query is None:
        raise ValueError("oceanrl not found! Cannot run optimization.")
    
    if study_name is None:
        study_name = f"{algo}_fishing_optimization"
    
    print("\n" + "="*70)
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION - {algo.upper()}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Algorithm: {algo}")
    print(f"  Trials: {n_trials}")
    print(f"  Timesteps per trial: {n_timesteps:,}")
    print(f"  Eval episodes: {n_eval_episodes}")
    print(f"  Device: {device}\n")
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=n_timesteps // 3)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial, algo, n_timesteps, n_eval_episodes, seed, device
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    os.makedirs("./optuna_results", exist_ok=True)
    results_file = f"./optuna_results/{algo}_best_params.json"
    with open(results_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\n💾 Best parameters saved to: {results_file}")
    
    # Save study
    study_file = f"./optuna_results/{algo}_study.pkl"
    import pickle
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    
    print(f"💾 Study saved to: {study_file}\n")
    
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal Optuna Hyperparameter Tuning"
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["sac", "td3", "tqc", "ppo"],
        help="RL algorithm to tune"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Timesteps per trial"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Episodes for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    run_optimization(
        algo=args.algo,
        n_trials=args.trials,
        n_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device
    )
