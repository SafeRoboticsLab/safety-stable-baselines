#!/usr/bin/env python3
"""
Script to load and replay a trained SafetySAC model in the Circle environment with visual rendering.
"""

import os
import sys
import argparse
import time
import numpy as np

from safety_gymnasium.safety_envs.safety_goal_margin import make_env

# Add parent directory to path so we can import safety_sb3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety_sb3 import SafetySAC


def replay_model(model_path: str, agent: str = "Car", level: int = 2, 
                 safety_clearance: float = 0.5, num_episodes: int = 5, 
                 deterministic: bool = True, step_delay: float = 0.02,
                 fixed_position: tuple = None, difficulty: str = None, position_index: int = 0):
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
        fixed_position: (x, y) tuple for fixed starting position. Overrides difficulty.
        difficulty: Predefined difficulty ("near_border", "very_close_to_border", "almost_unsafe")
        position_index: Index for predefined positions (0-3 for different sides)
    """
    print(f"Loading model from: {model_path}")
    
    # Determine position to use
    position = fixed_position
    if position is None and difficulty is not None:
        position = get_challenging_position(difficulty, level, position_index)

    # Create environment with rendering
    env = make_env(
        agent=agent, 
        level=2, 
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
    
    print(f"\n   Starting replay with:")
    print(f"   Agent: {agent}")
    print(f"   Level: {level}")
    print(f"   Safety clearance: {safety_clearance}m")
    print(f"   Episodes: {num_episodes}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Step delay: {step_delay}s")
    if position:
        print(f"   Fixed position: ({position[0]:.2f}, {position[1]:.2f})")
        if difficulty:
            print(f"   Difficulty: {difficulty} (index {position_index})")
    else:
        print(f"   Position: Random")
    print(f"\n   Goal: Navigate around the circle while staying safe!")
    print(f"   Margin function: g(s) = min_distance - {safety_clearance}")
    print(f"   Episode ends if g(s) < 0 (safety violation)\n")
    
    episode_stats = []
    
    try:
        for episode in range(num_episodes):
            print(f"   Episode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            env.render()
            episode_reward = 0.0
            episode_steps = 0
            safety_violations = 0
            min_margin = float('inf')
            q_values = []  # Track Q-values throughout the episode

            # # Keyboard input to choose initial state
            # print(f"   Initial state loaded for episode {episode + 1}")
            # print(f"   Press 'r' + Enter to reset and try a different initial state")
            # print(f"   Press any other key + Enter to run this episode")
            
            # try:
            #     user_input = input("Your choice: ").strip().lower()
            #     if user_input == 'r':
            #         print("   Resetting to get a different initial state...")
            #         continue  # Skip to next iteration of for loop
            #     else:
            #         print("   Running this episode...")
            # except KeyboardInterrupt:
            #     print(f"\n   Replay interrupted by user")
            #     return
            
            while True:
                # Get action from trained model
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Get Q-values for the current state-action pair
                try:
                    # Convert observation and action to tensor format
                    import torch
                    
                    # For SafetySAC, we need to access the critic differently
                    # SafetySAC has a different critic structure than standard SAC
                    device = next(model.critic.parameters()).device
                    
                    # Convert to tensors and move to device
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
                    
                    # Get Q-value from SafetySAC critic (expects separate obs and action)
                    with torch.no_grad():
                        q_value_output = model.critic(obs_tensor, action_tensor)
                        
                        # Handle different critic output formats
                        if isinstance(q_value_output, tuple):
                            # If tuple, take minimum (conservative)
                            q_value = torch.min(q_value_output[0], q_value_output[1]).cpu().numpy().item()
                        else:
                            # Single output
                            q_value = q_value_output.cpu().numpy().item()
                    
                except Exception as e:
                    # Fallback if Q-value computation fails
                    q_value = "N/A"
                    print(f"Debug: Q-value computation failed: {e}")  # Temporary debug line
                
                # Store Q-value for statistics
                if isinstance(q_value, float):
                    q_values.append(q_value)
                
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
                
                # Print step information including Q-value
                # if episode_steps % 10 == 0 or episode_steps <= 5:  # Print every 10 steps or first 5 steps
                if episode_steps % 1 == 0:
                    q_value_str = f"{q_value:.3f}" if isinstance(q_value, float) else str(q_value)
                    margin_str = f"{info.get('margin_g', 'N/A'):.3f}" if 'margin_g' in info else 'N/A'
                    print(f"      Step {episode_steps}: Q-value = {q_value_str}, "
                          f"Reward = {reward:.3f}, Margin = {margin_str}")
                
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
                'truncated': truncated,
                'avg_q_value': np.mean(q_values) if q_values else 'N/A',
                'min_q_value': np.min(q_values) if q_values else 'N/A',
                'max_q_value': np.max(q_values) if q_values else 'N/A'
            }
            episode_stats.append(final_info)
            
            print(f"      Episode {episode + 1} complete!")
            print(f"      Reward: {episode_reward:.3f}")
            print(f"      Steps: {episode_steps}")
            print(f"      Min margin: {final_info['min_margin']}")
            print(f"      Safety violations: {safety_violations}")
            avg_q_str = f"{final_info['avg_q_value']:.3f}" if isinstance(final_info['avg_q_value'], float) else str(final_info['avg_q_value'])
            min_q_str = f"{final_info['min_q_value']:.3f}" if isinstance(final_info['min_q_value'], float) else str(final_info['min_q_value'])
            max_q_str = f"{final_info['max_q_value']:.3f}" if isinstance(final_info['max_q_value'], float) else str(final_info['max_q_value'])
            print(f"      Q-values: avg={avg_q_str}, min={min_q_str}, max={max_q_str}")
            print(f"      Reason: {'Safety violation' if terminated else 'Timeout' if truncated else 'Complete'}")
            print()
            
            # Brief pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"\n   Replay interrupted by user")
    
    finally:
        env.close()
        
        # Print overall statistics
        if episode_stats:
            print(f"\n   Overall Statistics ({len(episode_stats)} episodes):")
            avg_reward = sum(ep['reward'] for ep in episode_stats) / len(episode_stats)
            avg_steps = sum(ep['steps'] for ep in episode_stats) / len(episode_stats)
            total_violations = sum(ep['safety_violations'] for ep in episode_stats)
            
            # Calculate overall Q-value statistics
            all_q_avg = [ep['avg_q_value'] for ep in episode_stats if isinstance(ep['avg_q_value'], float)]
            overall_avg_q = np.mean(all_q_avg) if all_q_avg else 'N/A'
            
            all_q_min = [ep['min_q_value'] for ep in episode_stats if isinstance(ep['min_q_value'], float)]
            overall_min_q = np.min(all_q_min) if all_q_min else 'N/A'
            
            all_q_max = [ep['max_q_value'] for ep in episode_stats if isinstance(ep['max_q_value'], float)]
            overall_max_q = np.max(all_q_max) if all_q_max else 'N/A'
            
            print(f"   Average reward: {avg_reward:.3f}")
            print(f"   Average steps: {avg_steps:.1f}")
            print(f"   Total safety violations: {total_violations}")
            
            overall_avg_q_str = f"{overall_avg_q:.3f}" if isinstance(overall_avg_q, float) else str(overall_avg_q)
            overall_min_q_str = f"{overall_min_q:.3f}" if isinstance(overall_min_q, float) else str(overall_min_q)
            overall_max_q_str = f"{overall_max_q:.3f}" if isinstance(overall_max_q, float) else str(overall_max_q)
            print(f"   Q-values: avg={overall_avg_q_str}, min={overall_min_q_str}, max={overall_max_q_str}")
            
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
    parser.add_argument("--safety-clearance", "-s", type=float, default=0.0,
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
        print(f"   Model file not found: {args.model}")
        print("Make sure you have trained a model first using one of the training scripts.")
        return
        
    replay_model(
        model_path=args.model,
        agent=args.agent,
        level=args.level,
        safety_clearance=args.safety_clearance,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        step_delay=args.delay,
    )


if __name__ == "__main__":
    main()