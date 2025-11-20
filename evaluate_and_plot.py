"""
Universal Evaluation and Visualization Script
==============================================

Evaluates any trained RL agent and generates comprehensive plots:
1. Population dynamics over time (Salmon & Sharks)
2. Fishing effort over time
3. Catches per month
4. Cumulative rewards
5. Episode return breakdown

Works with: SAC, TD3, TQC, PPO

Usage:
    python evaluate_and_plot.py --algo sac --model models/sac_fishing_model.zip
    python evaluate_and_plot.py --algo td3 --episodes 5
    python evaluate_and_plot.py --algo tqc --model models/tqc_fishing_model.zip --save-data

Author: AY25/26 T1 Project
Date: November 2025
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import pandas as pd
import os
from datetime import datetime

try:
    from oceanrl import query as ocean_query
    print("✅ oceanrl imported")
except ImportError:
    print("❌ oceanrl not found!")
    ocean_query = None

# Constants
K1, K2, K3, K4 = 0.001, 0.01, 100.0, 100.0
EPISODE_LENGTH_MONTHS = 900


def load_agent(algo: str, model_path: str):
    """Load trained agent based on algorithm type"""
    
    if algo == "sac":
        from sac_agent import SACFishingAgent
        agent = SACFishingAgent(model_path=model_path)
    elif algo == "td3":
        from td3_agent import TD3FishingAgent
        agent = TD3FishingAgent(model_path=model_path)
    elif algo == "tqc":
        from tqc_agent import TQCFishingAgent
        agent = TQCFishingAgent(model_path=model_path)
    elif algo == "ppo":
        from ppo_agent import PPOFishingAgent
        agent = PPOFishingAgent(model_path=model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return agent


def run_episode(agent, initial_salmon: float = 20000.0, initial_shark: float = 500.0) -> Dict:
    """Run one episode and collect detailed statistics"""
    
    salmon_t = initial_salmon
    shark_t = initial_shark
    
    history = {
        'month': [],
        'salmon': [],
        'shark': [],
        'effort': [],
        'caught': [],
        'immediate_reward': [],
        'cumulative_reward': []
    }
    
    cumulative_reward = 0.0
    total_caught = 0.0
    total_effort = 0.0
    
    for month_t in range(1, EPISODE_LENGTH_MONTHS + 1):
        # Get agent's action
        effort = agent.act((salmon_t, shark_t, month_t))
        
        # Query ecosystem
        caught, next_salmon, next_shark = ocean_query(
            salmon_t, shark_t, effort, month_t
        )
        
        # Calculate immediate reward
        immediate_reward = K1 * caught - K2 * effort
        cumulative_reward += immediate_reward
        
        # Store data
        history['month'].append(month_t)
        history['salmon'].append(salmon_t)
        history['shark'].append(shark_t)
        history['effort'].append(effort)
        history['caught'].append(caught)
        history['immediate_reward'].append(immediate_reward)
        history['cumulative_reward'].append(cumulative_reward)
        
        # Update totals
        total_caught += caught
        total_effort += effort
        
        # Update state
        salmon_t = next_salmon
        shark_t = next_shark
    
    # Add terminal bonus
    terminal_bonus = (
        K3 * math.log(max(1e-10, salmon_t)) +
        K4 * math.log(max(1e-10, shark_t))
    )
    
    total_return = cumulative_reward + terminal_bonus
    
    # Summary statistics
    summary = {
        'total_return': total_return,
        'cumulative_reward': cumulative_reward,
        'terminal_bonus': terminal_bonus,
        'total_caught': total_caught,
        'total_effort': total_effort,
        'final_salmon': salmon_t,
        'final_shark': shark_t,
        'avg_effort': total_effort / EPISODE_LENGTH_MONTHS,
        'avg_catch': total_caught / EPISODE_LENGTH_MONTHS
    }
    
    return history, summary


def plot_results(
    histories: List[Dict],
    summaries: List[Dict],
    algo: str,
    save_dir: str = "./evaluation_results"
):
    """Generate comprehensive plots"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # If multiple episodes, average them
    if len(histories) > 1:
        # Convert to DataFrame for easy averaging
        dfs = [pd.DataFrame(h) for h in histories]
        avg_history = pd.concat(dfs).groupby('month').mean().reset_index()
        std_history = pd.concat(dfs).groupby('month').std().reset_index()
    else:
        avg_history = pd.DataFrame(histories[0])
        std_history = None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ====================================================================
    # 1. POPULATION DYNAMICS (Main Plot - Large)
    # ====================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    months = avg_history['month']
    
    # Plot salmon
    ax1.plot(months, avg_history['salmon'], 
             color='#FF6B6B', linewidth=2.5, label='Salmon', alpha=0.9)
    if std_history is not None:
        ax1.fill_between(months, 
                        avg_history['salmon'] - std_history['salmon'],
                        avg_history['salmon'] + std_history['salmon'],
                        color='#FF6B6B', alpha=0.2)
    
    # Plot sharks on secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(months, avg_history['shark'], 
                  color='#4ECDC4', linewidth=2.5, label='Sharks', alpha=0.9)
    if std_history is not None:
        ax1_twin.fill_between(months,
                             avg_history['shark'] - std_history['shark'],
                             avg_history['shark'] + std_history['shark'],
                             color='#4ECDC4', alpha=0.2)
    
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Salmon Population', fontsize=12, fontweight='bold', color='#FF6B6B')
    ax1_twin.set_ylabel('Shark Population', fontsize=12, fontweight='bold', color='#4ECDC4')
    ax1.set_title(f'Population Dynamics - {algo.upper()}', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#FF6B6B')
    ax1_twin.tick_params(axis='y', labelcolor='#4ECDC4')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # ====================================================================
    # 2. SUMMARY STATISTICS (Top Right)
    # ====================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Calculate average statistics
    avg_summary = {k: np.mean([s[k] for s in summaries]) for k in summaries[0].keys()}
    std_summary = {k: np.std([s[k] for s in summaries]) for k in summaries[0].keys()}
    
    summary_text = f"""
    EPISODE SUMMARY
    {'='*30}
    
    Total Return:
      {avg_summary['total_return']:,.2f} ± {std_summary['total_return']:.2f}
    
    Components:
      Fishing Reward: {avg_summary['cumulative_reward']:,.2f}
      Terminal Bonus: {avg_summary['terminal_bonus']:,.2f}
    
    Fishing Performance:
      Total Caught: {avg_summary['total_caught']:,.0f}
      Total Effort: {avg_summary['total_effort']:,.2f}
      Avg Catch/Month: {avg_summary['avg_catch']:,.1f}
      Avg Effort/Month: {avg_summary['avg_effort']:,.2f}
    
    Final Populations:
      Salmon: {avg_summary['final_salmon']:,.0f}
      Sharks: {avg_summary['final_shark']:,.0f}
    
    Episodes Evaluated: {len(summaries)}
    """
    
    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ====================================================================
    # 3. FISHING EFFORT OVER TIME
    # ====================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(months, avg_history['effort'], color='#95E1D3', linewidth=2)
    ax3.fill_between(months, avg_history['effort'], alpha=0.4, color='#95E1D3')
    
    if std_history is not None:
        ax3.fill_between(months,
                        avg_history['effort'] - std_history['effort'],
                        avg_history['effort'] + std_history['effort'],
                        color='#95E1D3', alpha=0.2)
    
    ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Fishing Effort', fontsize=11, fontweight='bold')
    ax3.set_title('Fishing Effort Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ====================================================================
    # 4. SALMON CAUGHT PER MONTH
    # ====================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(months, avg_history['caught'], color='#F38181', linewidth=2)
    ax4.fill_between(months, avg_history['caught'], alpha=0.4, color='#F38181')
    
    if std_history is not None:
        ax4.fill_between(months,
                        avg_history['caught'] - std_history['caught'],
                        avg_history['caught'] + std_history['caught'],
                        color='#F38181', alpha=0.2)
    
    ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Salmon Caught', fontsize=11, fontweight='bold')
    ax4.set_title('Monthly Harvest', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ====================================================================
    # 5. CUMULATIVE REWARD
    # ====================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    ax5.plot(months, avg_history['cumulative_reward'], 
             color='#AA96DA', linewidth=2.5)
    
    if std_history is not None:
        ax5.fill_between(months,
                        avg_history['cumulative_reward'] - std_history['cumulative_reward'],
                        avg_history['cumulative_reward'] + std_history['cumulative_reward'],
                        color='#AA96DA', alpha=0.2)
    
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Cumulative Reward', fontsize=11, fontweight='bold')
    ax5.set_title('Reward Accumulation', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ====================================================================
    # 6. IMMEDIATE REWARDS OVER TIME
    # ====================================================================
    ax6 = fig.add_subplot(gs[2, 0])
    
    ax6.plot(months, avg_history['immediate_reward'], 
             color='#FFA07A', linewidth=1.5, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Immediate Reward', fontsize=11, fontweight='bold')
    ax6.set_title('Per-Step Rewards', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # ====================================================================
    # 7. POPULATION RATIO (Salmon/Shark)
    # ====================================================================
    ax7 = fig.add_subplot(gs[2, 1])
    
    ratio = avg_history['salmon'] / (avg_history['shark'] + 1e-10)
    ax7.plot(months, ratio, color='#9B59B6', linewidth=2)
    ax7.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Salmon/Shark Ratio', fontsize=11, fontweight='bold')
    ax7.set_title('Ecosystem Balance', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # ====================================================================
    # 8. CATCH EFFICIENCY (Catch/Effort Ratio)
    # ====================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    
    efficiency = avg_history['caught'] / (avg_history['effort'] + 1e-10)
    ax8.plot(months, efficiency, color='#E67E22', linewidth=2)
    ax8.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Catch per Unit Effort', fontsize=11, fontweight='bold')
    ax8.set_title('Fishing Efficiency', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # ====================================================================
    # SAVE FIGURE
    # ====================================================================
    plt.suptitle(f'{algo.upper()} Agent - Evaluation Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/{algo}_evaluation_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {filename}")
    
    plt.close()


def save_data(histories: List[Dict], summaries: List[Dict], algo: str, save_dir: str):
    """Save evaluation data to CSV files"""
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save episode histories
    for i, history in enumerate(histories):
        df = pd.DataFrame(history)
        filename = f"{save_dir}/{algo}_episode_{i+1}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Data saved: {filename}")
    
    # Save summary statistics
    summary_df = pd.DataFrame(summaries)
    summary_filename = f"{save_dir}/{algo}_summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"💾 Summary saved: {summary_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize RL agent performance"
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["sac", "td3", "tqc", "ppo"],
        help="Algorithm to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: ./models/{algo}_fishing_model.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to evaluate (default: 3)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results (default: ./evaluation_results)"
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save raw data to CSV files"
    )
    parser.add_argument(
        "--initial-salmon",
        type=float,
        default=20000.0,
        help="Initial salmon population (default: 20000)"
    )
    parser.add_argument(
        "--initial-shark",
        type=float,
        default=500.0,
        help="Initial shark population (default: 500)"
    )
    
    args = parser.parse_args()
    
    # Check oceanrl
    if ocean_query is None:
        print("❌ ERROR: oceanrl not found!")
        print("Install: pip install oceanrl-0.1.0-py3-none-any.whl")
        return
    
    # Default model path
    if args.model is None:
        args.model = f"./models/{args.algo}_fishing_model.zip"
    
    print("\n" + "="*70)
    print(f"EVALUATING {args.algo.upper()} AGENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Initial Salmon: {args.initial_salmon:,.0f}")
    print(f"  Initial Sharks: {args.initial_shark:,.0f}")
    print(f"  Save directory: {args.save_dir}")
    print()
    
    # Load agent
    try:
        agent = load_agent(args.algo, args.model)
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return
    
    # Run episodes
    histories = []
    summaries = []
    
    for ep in range(args.episodes):
        print(f"Running episode {ep + 1}/{args.episodes}...")
        history, summary = run_episode(
            agent,
            initial_salmon=args.initial_salmon,
            initial_shark=args.initial_shark
        )
        histories.append(history)
        summaries.append(summary)
        
        print(f"  Episode {ep + 1} Return: {summary['total_return']:,.2f}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    returns = [s['total_return'] for s in summaries]
    print(f"\nReturns across {args.episodes} episodes:")
    print(f"  Mean: {np.mean(returns):,.2f}")
    print(f"  Std:  {np.std(returns):,.2f}")
    print(f"  Min:  {np.min(returns):,.2f}")
    print(f"  Max:  {np.max(returns):,.2f}")
    
    avg_summary = {k: np.mean([s[k] for s in summaries]) for k in summaries[0].keys()}
    print(f"\nAverage Statistics:")
    print(f"  Total Caught: {avg_summary['total_caught']:,.0f}")
    print(f"  Total Effort: {avg_summary['total_effort']:,.2f}")
    print(f"  Final Salmon: {avg_summary['final_salmon']:,.0f}")
    print(f"  Final Sharks: {avg_summary['final_shark']:,.0f}")
    print()
    
    # Generate plots
    print("📊 Generating plots...")
    plot_results(histories, summaries, args.algo, args.save_dir)
    
    # Save data if requested
    if args.save_data:
        print("💾 Saving data files...")
        save_data(histories, summaries, args.algo, args.save_dir)
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved in: {args.save_dir}/")
    print()


if __name__ == "__main__":
    main()
