#!/usr/bin/env python3
"""
Generate a dataset of initial states for consistent evaluation.

This script creates a pickle file containing N initial environment states
that can be used to ensure consistent evaluation across different models.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import safety_gymnasium
from datetime import datetime
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_initial_states(env_id: str = "SafetyCarCircle2-v0", 
                           num_states: int = 100,
                           output_file: str = None,
                           seed: int = None):
    """
    Generate a dataset of initial environment states.
    
    Args:
        env_id: Safety-gymnasium environment ID
        num_states: Number of initial states to generate
        output_file: Path to save the pickle file (default: auto-generated)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        Path to the saved pickle file
    """
    print(f"Generating {num_states} initial states for {env_id}")
    
    # Create environment
    env = safety_gymnasium.make(env_id, render_mode=None)
    env = TerminateOnCollisionWrapper(env)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    
    # Set seed if provided
    if seed is not None:
        print(f"Using random seed: {seed}")
        np.random.seed(seed)
    
    # Generate initial states
    initial_states = []
    
    print("\nGenerating states...")
    for i in range(num_states):
        # Reset environment to get a new initial state
        obs, info = env.reset()
        
        # Get the full environment state using World API
        # This includes time, qpos, qvel, and act (actuator state)
        if hasattr(env.unwrapped, 'task') and hasattr(env.unwrapped.task, 'world'):
            # Safety Gymnasium environment - use World API
            world = env.unwrapped.task.world
            mujoco_state = world.get_state()  # Returns {'time', 'qpos', 'qvel', 'act' or None}
            
            state = {
                'mujoco_state': mujoco_state,
                'observation': obs.copy(),
                'info': info.copy() if isinstance(info, dict) else {}
            }
        else:
            # Fallback - just save observation
            print("Warning: Could not access World API, saving observation only")
            state = {
                'observation': obs.copy(),
                'info': info.copy() if isinstance(info, dict) else {}
            }
        
        initial_states.append(state)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_states} states")
    
    env.close()
    
    # Create output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "initial_states"
        os.makedirs(output_dir, exist_ok=True)
        
        seed_str = f"_seed{seed}" if seed is not None else ""
        output_file = os.path.join(
            output_dir, 
            f"{env_id}_n{num_states}{seed_str}_{timestamp}.pkl"
        )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to pickle file
    dataset = {
        'env_id': env_id,
        'num_states': num_states,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'states': initial_states
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nSaved {num_states} initial states to: {output_file}")
    
    # Print statistics
    if initial_states and 'mujoco_state' in initial_states[0]:
        mujoco_state = initial_states[0]['mujoco_state']
        print(f"\nState information:")
        print(f"  time: {mujoco_state.get('time', 'N/A')}")
        print(f"  qpos dimension: {len(mujoco_state['qpos']) if 'qpos' in mujoco_state else 'N/A'}")
        print(f"  qvel dimension: {len(mujoco_state['qvel']) if 'qvel' in mujoco_state else 'N/A'}")
        print(f"  act: {'Yes' if mujoco_state.get('act') is not None else 'None'}")
        print(f"  observation dimension: {len(initial_states[0]['observation'])}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset of initial environment states for consistent evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 initial states for CarCircle2
  python generate_initial_states.py --env SafetyCarCircle2-v0 --num-states 100
  
  # Generate with specific seed for reproducibility
  python generate_initial_states.py --env SafetyCarCircle2-v0 --num-states 100 --seed 42
  
  # Generate and save to specific file
  python generate_initial_states.py --env SafetyCarCircle2-v0 --num-states 50 \\
      --output initial_states/car_circle_eval_states.pkl
  
  # Generate states for different environment
  python generate_initial_states.py --env SafetyCarGoal2-v0 --num-states 100 --seed 42
        """
    )
    
    parser.add_argument("--env", type=str, default="SafetyCarCircle2-v0",
                        choices=["SafetyCarCircle1-v0", "SafetyCarCircle2-v0",
                                "SafetyCarGoal1-v0", "SafetyCarGoal2-v0",
                                "SafetyPointCircle1-v0", "SafetyPointCircle2-v0",
                                "SafetyPointGoal1-v0", "SafetyPointGoal2-v0"],
                        help="Safety-gymnasium environment ID")
    parser.add_argument("--num-states", "-n", type=int, default=100,
                        help="Number of initial states to generate (default: 100)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output pickle file path (default: auto-generated)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    
    args = parser.parse_args()
    
    generate_initial_states(
        env_id=args.env,
        num_states=args.num_states,
        output_file=args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
