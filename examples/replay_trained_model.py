#!/usr/bin/env python3
"""
Script to load and replay a trained SafetySAC model in the Circle environment with visual rendering.
"""

import os
import sys
import argparse
import time

from safety_gymnasium.safety_envs.safety_circle_margin import make_env

# Add parent directory to path so we can import safety_sb3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety_sb3 import SafetySAC


def replay_model(model_path: str, agent: str = "Car", level: int = 2, 
                 safety_clearance: float = 0.5, num_episodes: int = 5, 
                 deterministic: bool = True, step_delay: float = 0.02):
    """
    Load and replay a trained model.
    
    Args:
        model_path: Path to the saved model (.zip file)
        agent: Agent type (Car, Point, etc.)
        level: Circle environment level (0, 1, 2)
        safety_clearance: Safety margin in meters
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions (True) or sample from policy (False)
        step_delay: Delay between steps for better visualization (seconds)
    """
    print(f"Loading model from: {model_path}")
    
    # Create environment with rendering
    env = make_env(
        agent=agent, 
        level=level, 
        render_mode="human", 
        safety_clearance=safety_clearance
    )
    
    # Load the trained model
    try:
        model = SafetySAC.load(model_path, env=env)
        print(f"✓ Successfully loaded SafetySAC model")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Make sure the model path is correct and the model was saved properly.")
        return
    
    print(f"\n🎮 Starting replay with:")
    print(f"   Agent: {agent}")
    print(f"   Level: {level}")
    print(f"   Safety clearance: {safety_clearance}m")
    print(f"   Episodes: {num_episodes}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Step delay: {step_delay}s")
    print(f"\n🎯 Goal: Navigate around the circle while staying safe!")
    print(f"   Margin function: g(s) = min_distance - {safety_clearance}")
    print(f"   Episode ends if g(s) < 0 (safety violation)\n")
    
    episode_stats = []
    
    try:
        for episode in range(num_episodes):
            print(f"🚀 Episode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            safety_violations = 0
            min_margin = float('inf')
            
            while True:
                # Get action from trained model
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track statistics
                episode_reward += reward
                episode_steps += 1
                
                # Track safety margin if available
                if 'margin_g' in info:
                    margin = info['margin_g']
                    min_margin = min(min_margin, margin)
                    if margin < 0:
                        safety_violations += 1
                
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
                'steps': episode_steps,
                'min_margin': min_margin if min_margin != float('inf') else 'N/A',
                'safety_violations': safety_violations,
                'terminated': terminated,
                'truncated': truncated
            }
            episode_stats.append(final_info)
            
            print(f"   ✅ Episode {episode + 1} complete!")
            print(f"      Reward: {episode_reward:.3f}")
            print(f"      Steps: {episode_steps}")
            print(f"      Min margin: {final_info['min_margin']}")
            print(f"      Safety violations: {safety_violations}")
            print(f"      Reason: {'Safety violation' if terminated else 'Timeout' if truncated else 'Complete'}")
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
            avg_steps = sum(ep['steps'] for ep in episode_stats) / len(episode_stats)
            total_violations = sum(ep['safety_violations'] for ep in episode_stats)
            
            print(f"   Average reward: {avg_reward:.3f}")
            print(f"   Average steps: {avg_steps:.1f}")
            print(f"   Total safety violations: {total_violations}")
            
            # Safety rate
            safe_episodes = sum(1 for ep in episode_stats if ep['safety_violations'] == 0)
            safety_rate = safe_episodes / len(episode_stats) * 100
            print(f"   Safety rate: {safety_rate:.1f}% ({safe_episodes}/{len(episode_stats)} episodes)")


def main():
    parser = argparse.ArgumentParser(description="Replay a trained SafetySAC model")
    parser.add_argument("--model", "-m", type=str, default="checkpoints/safety_sac_car_circle2_70000_steps.zip",
                        help="Path to the trained model file")
    parser.add_argument("--agent", "-a", type=str, default="Car", 
                        choices=["Point", "Car", "Racecar", "Doggo", "Ant"],
                        help="Agent type")
    parser.add_argument("--level", "-l", type=int, default=2, choices=[0, 1, 2],
                        help="Circle environment level")
    parser.add_argument("--safety-clearance", "-s", type=float, default=0.5,
                        help="Safety clearance in meters")
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
        print("Make sure you have trained a model first using one of the training scripts.")
        return
    
    replay_model(
        model_path=args.model,
        agent=args.agent,
        level=args.level,
        safety_clearance=args.safety_clearance,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        step_delay=args.delay
    )


if __name__ == "__main__":
    main()