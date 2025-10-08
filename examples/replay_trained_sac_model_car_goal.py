#!/usr/bin/env python3
"""
Script to load and replay a trained SAC model in the original safety-gymnasium CarGoal environment.
"""

import os
import sys
import argparse
import time
import safety_gymnasium

from stable_baselines3 import SAC
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def replay_sac_model(model_path: str, env_id: str = "SafetyCarGoal2-v0", 
                     num_episodes: int = 5, deterministic: bool = True, 
                     step_delay: float = 0.02):
    """
    Load and replay a trained SAC model.
    
    Args:
        model_path: Path to the saved model (.zip file)
        env_id: Safety-gymnasium environment ID
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions (True) or sample from policy (False)
        step_delay: Delay between steps for better visualization (seconds)
    """
    print(f"Loading SAC model from: {model_path}")
    
    # Create original safety-gymnasium environment with rendering
    env = safety_gymnasium.make(env_id, render_mode="human")
    env = TerminateOnCollisionWrapper(env)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    
    # Load the trained SAC model
    try:
        model = SAC.load(model_path, env=env)
        print(f"✓ Successfully loaded SAC model")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Make sure the model path is correct and the model was saved properly.")
        return
    
    print(f"\n🎮 Starting SAC replay with:")
    print(f"   Environment: {env_id}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Step delay: {step_delay}s")
    print(f"\n🎯 Goal: Navigate using original safety-gymnasium rewards and constraints!")
    print(f"   This uses the original environment without margin-based modifications")
    print(f"   Safety violations will be tracked via environment's built-in cost signal\n")
    
    episode_stats = []
    
    try:
        for episode in range(num_episodes):
            print(f"🚀 Episode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0  # Track safety violations via cost
            episode_steps = 0
            cost_violations = 0
            
            while True:
                # Get action from trained SAC model
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track statistics
                episode_reward += reward
                episode_steps += 1
                
                # Track cost/safety violations if available
                if 'cost' in info:
                    cost = info['cost']
                    episode_cost += cost
                    if cost > 0:
                        cost_violations += 1
                
                # Render the environment
                env.render()
                
                # Add delay for better visualization
                if step_delay > 0:
                    time.sleep(step_delay)
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Episode summary
            final_info = {
                'episode': episode + 1,
                'reward': episode_reward,
                'cost': episode_cost,
                'steps': episode_steps,
                'cost_violations': cost_violations,
                'terminated': terminated,
                'truncated': truncated
            }
            episode_stats.append(final_info)
            
            print(f"   ✅ Episode {episode + 1} complete!")
            print(f"      Reward: {episode_reward:.3f}")
            print(f"      Cost: {episode_cost:.3f}")
            print(f"      Steps: {episode_steps}")
            print(f"      Cost violations: {cost_violations}")
            print(f"      Reason: {'Task complete/Safety violation' if terminated else 'Timeout' if truncated else 'Complete'}")
            print()
            
            # Brief pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Replay interrupted by user")
    
    finally:
        env.close()
        
        # Print overall statistics
        if episode_stats:
            print(f"\n📊 Overall Statistics ({len(episode_stats)} episodes):")
            avg_reward = sum(ep['reward'] for ep in episode_stats) / len(episode_stats)
            avg_cost = sum(ep['cost'] for ep in episode_stats) / len(episode_stats)
            avg_steps = sum(ep['steps'] for ep in episode_stats) / len(episode_stats)
            total_violations = sum(ep['cost_violations'] for ep in episode_stats)
            
            print(f"   Average reward: {avg_reward:.3f}")
            print(f"   Average cost: {avg_cost:.3f}")
            print(f"   Average steps: {avg_steps:.1f}")
            print(f"   Total cost violations: {total_violations}")
            
            # Safety rate (episodes with zero cost)
            safe_episodes = sum(1 for ep in episode_stats if ep['cost'] == 0)
            safety_rate = safe_episodes / len(episode_stats) * 100
            print(f"   Safety rate: {safety_rate:.1f}% ({safe_episodes}/{len(episode_stats)} episodes with zero cost)")


def main():
    parser = argparse.ArgumentParser(description="Replay a trained SAC model on safety-gymnasium")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to the trained SAC model file (.zip)")
    parser.add_argument("--env", type=str, default="SafetyCarGoal1-v0",
                        choices=["SafetyCarGoal1-v0", "SafetyCarGoal2-v0"],
                        help="Safety-gymnasium environment ID")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--delay", "-d", type=float, default=0.02,
                        help="Delay between steps for visualization (seconds)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Make sure you have trained a SAC model first using car_goal_naive_train.py")
        print("or car_goal_naive_train_withFilter.py")
        return
    
    replay_sac_model(
        model_path=args.model,
        env_id=args.env,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        step_delay=args.delay
    )


if __name__ == "__main__":
    main()