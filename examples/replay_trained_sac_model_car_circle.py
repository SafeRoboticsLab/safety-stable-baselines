#!/usr/bin/env python3
"""
Script to load and replay a trained SAC model in the original safety-gymnasium Circle environment.
Supports testing with different safety filter configurations:
1. No filter (default)
2. Safety value filter (based on Q-value margin)
3. Safety rollout filter (based on rollout simulation)
"""

import os
import sys
import argparse
import time
import pickle
import safety_gymnasium
import torch
<<<<<<< HEAD
import mujoco
=======
>>>>>>> 8a47eebe1a66136e68569efc17a06b7d5cb8525a

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.observation_storing_wrapper import ObservationStoringWrapper
from wrappers.safety_rollout_filter_wrapper import SafetyRolloutFilter
from wrappers.safety_value_filter_wrapper import SafetyFilterWrapper
from wrappers.video_recording_wrapper import VideoRecordingWrapper


def print_model_architecture(model, env):
    """
    Print detailed architecture information of the SAC model.
    
    Args:
        model: Trained SAC model
        env: Environment (for getting observation/action space info)
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE INFORMATION")
    print("="*80)
    
    # Input/Output shapes
    print("\nINPUT/OUTPUT SHAPES:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Observation dimension: {env.observation_space.shape[0]}")
    print(f"   Action space: {env.action_space}")
    print(f"   Action shape: {env.action_space.shape}")
    print(f"   Action dimension: {env.action_space.shape[0]}")
    
    # Actor network
    print("\nACTOR NETWORK (Policy):")
    print(f"   Type: {type(model.actor).__name__}")
    print(f"   Input dim: {env.observation_space.shape[0]}")
    print(f"   Output dim: {env.action_space.shape[0]} (mean and log_std)")
    
    if hasattr(model.actor, 'latent_pi'):
        print(f"\n   Latent Policy Network:")
        for i, layer in enumerate(model.actor.latent_pi):
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                print(f"      Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
            else:
                print(f"      Layer {i}: {layer}")
    
    if hasattr(model.actor, 'mu'):
        print(f"   Mean output layer: {model.actor.mu}")
    if hasattr(model.actor, 'log_std'):
        print(f"   Log std output layer: {model.actor.log_std}")
    
    # Critic network
    print("\nCRITIC NETWORK (Q-function):")
    print(f"   Type: {type(model.critic).__name__}")
    print(f"   Number of critics: {model.critic.n_critics if hasattr(model.critic, 'n_critics') else 'N/A'}")
    print(f"   Input dim: {env.observation_space.shape[0]} (obs) + {env.action_space.shape[0]} (action)")
    print(f"   Output dim: 1 (Q-value)")
    
    if hasattr(model.critic, 'q_networks') and len(model.critic.q_networks) > 0:
        print(f"\n   Q-Network 1:")
        for i, layer in enumerate(model.critic.q_networks[0]):
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                print(f"      Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
            else:
                print(f"      Layer {i}: {layer}")
        
        if len(model.critic.q_networks) > 1:
            print(f"\n   Q-Network 2 (same architecture)")
    
    # Policy kwargs
    print("\nPOLICY CONFIGURATION:")
    if hasattr(model, 'policy_kwargs') and model.policy_kwargs:
        for key, value in model.policy_kwargs.items():
            print(f"   {key}: {value}")
    else:
        print("   Using default configuration")
    
    # Additional model parameters
    print("\nTRAINING PARAMETERS:")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Buffer size: {model.buffer_size}")
    print(f"   Batch size: {model.batch_size}")
    print(f"   Gamma (discount): {model.gamma}")
    print(f"   Tau (soft update): {model.tau}")
    if hasattr(model, 'ent_coef'):
        ent_coef = model.ent_coef
        if isinstance(ent_coef, torch.Tensor):
            ent_coef = ent_coef.item()
        print(f"   Entropy coefficient: {ent_coef}")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\nTOTAL PARAMETERS:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    print("="*80 + "\n")


def replay_sac_model(model_path: str, env_id: str = "SafetyCarCircle2-v0", 
                     num_episodes: int = 5, deterministic: bool = True, 
                     step_delay: float = 0.02, filter_mode: str = "none",
                     safety_model_path: str = None, filter_epsilon: float = 0.0,
                     rollout_horizon: int = 10, rollout_velocity_threshold: float = 0.1,
                     record_video: bool = False, video_folder: str = "./videos",
                     video_fps: int = 30, video_camera: str = None,
                     initial_states_file: str = None, run_index: int = 0):
    """
    Load and replay a trained SAC model with optional safety filters.
    
    Args:
        model_path: Path to the saved model (.zip file)
        env_id: Safety-gymnasium environment ID
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions (True) or sample from policy (False)
        step_delay: Delay between steps for better visualization (seconds)
        filter_mode: Safety filter mode - "none", "value", or "rollout"
        safety_model_path: Path to trained SafetySAC model (required for filters)
        filter_epsilon: Safety margin threshold for value filter
        rollout_horizon: Rollout horizon for rollout filter
        rollout_velocity_threshold: Velocity threshold for rollout filter
        record_video: Whether to record videos of episodes
        video_folder: Directory to save video recordings
        video_fps: Frame rate for video recordings
        video_camera: Camera name for video recording (e.g., 'vision', 'track', 'fixednear', 'fixedfar')
        initial_states_file: Path to pickle file with initial states (default: None)
        run_index: Starting index in initial states dataset (default: 0)
    """
    print(f"Loading SAC model from: {model_path}")
    
    # Load initial states if provided
    initial_states = None
    if initial_states_file is not None:
        if not os.path.exists(initial_states_file):
            print(f"Error: Initial states file not found: {initial_states_file}")
            return
        
        print(f"Loading initial states from: {initial_states_file}")
        with open(initial_states_file, 'rb') as f:
            dataset = pickle.load(f)
        
        initial_states = dataset['states']
        print(f"  Loaded {len(initial_states)} initial states")
        print(f"  Starting from index: {run_index}")
        
        if run_index >= len(initial_states):
            print(f"Error: run_index ({run_index}) >= number of states ({len(initial_states)})")
            return
        
        if run_index + num_episodes > len(initial_states):
            print(f"Warning: Not enough states for all episodes. Will run {len(initial_states) - run_index} episodes.")
            num_episodes = len(initial_states) - run_index
    
    # Create original safety-gymnasium environment with rendering
    # Use rgb_array for video recording, or human for visualization only
    render_mode = "rgb_array" if record_video else "human"
    env = safety_gymnasium.make(env_id, 
        render_mode=render_mode,
        width=1920,
        height=1080,
        camera_name=video_camera
    )
    env = TerminateOnCollisionWrapper(env)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    
    # Apply safety filter if requested
    if filter_mode == "value":
        if safety_model_path is None:
            print("Error: --safety-model is required for value filter mode")
            return
        if not os.path.exists(safety_model_path):
            print(f"Error: Safety model not found at {safety_model_path}")
            return
        print(f"\nApplying Safety Value Filter (epsilon={filter_epsilon})")
        env = ObservationStoringWrapper(env)
        env = SafetyFilterWrapper(env, safety_model_path, epsilon=filter_epsilon)
        env = Monitor(env)
        
    elif filter_mode == "rollout":
        if safety_model_path is None:
            print("Error: --safety-model is required for rollout filter mode")
            return
        if not os.path.exists(safety_model_path):
            print(f"Error: Safety model not found at {safety_model_path}")
            return
        print(f"\nApplying Safety Rollout Filter (horizon={rollout_horizon}, velocity_threshold={rollout_velocity_threshold})")
        env = ObservationStoringWrapper(env)
        env = SafetyRolloutFilter(env, safety_model_path, 
                                 horizon=rollout_horizon, 
                                 velocity_threshold=rollout_velocity_threshold)
        env = Monitor(env)
    
    # Add video recording wrapper if requested (before other wrappers)
    if record_video:
        # Extract model information from path for video naming
        # e.g., "experiments/20251008_0029_SAC_CarCircle2/best/best_model.zip"
        # -> "best_best_model" or "final_car_circle2" or "checkpoints_ckpt_123"
        model_info = "model"
        try:
            model_parts = model_path.replace('\\', '/').split('/')
            # Look for best/final/checkpoints in path
            if 'best' in model_parts:
                idx = model_parts.index('best')
                if idx + 1 < len(model_parts):
                    filename = model_parts[idx + 1].replace('.zip', '')
                    model_info = f"best_{filename}"
                else:
                    model_info = "best"
            elif 'final' in model_parts:
                idx = model_parts.index('final')
                if idx + 1 < len(model_parts):
                    filename = model_parts[idx + 1].replace('.zip', '')
                    model_info = f"final_{filename}"
                else:
                    model_info = "final"
            elif 'checkpoints' in model_parts or 'checkpoint' in model_parts:
                # Find the checkpoint part
                for part in reversed(model_parts):
                    if 'ckpt' in part.lower() or 'step' in part.lower():
                        model_info = f"ckpt_{part.replace('.zip', '')}"
                        break
                else:
                    model_info = "checkpoint"
            else:
                # Use the model filename
                model_info = model_parts[-1].replace('.zip', '')
        except Exception as e:
            print(f"Warning: Could not extract model info from path: {e}")
            model_info = "model"
        
        # Determine video name prefix based on filter mode and model info
        if filter_mode != "none":
            video_prefix = f"{filter_mode}_filter_{model_info}"
        else:
            video_prefix = f"no_filter_{model_info}"
        
        env = VideoRecordingWrapper(
            env, 
            video_folder=video_folder,
            name_prefix=video_prefix,
            fps=video_fps,
            record_every_n_episodes=1,  # Record all episodes
            include_stats_in_name=True
        )

    # Load the trained SAC model
    try:
        model = SAC.load(model_path, env=env)
        print(f"Successfully loaded SAC model")
        
        # Print architecture information
        print_model_architecture(model, env)
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure the model path is correct and the model was saved properly.")
        return
    
    print(f"\nStarting SAC replay with:")
    print(f"   Environment: {env_id}")
    print(f"   Filter mode: {filter_mode}")
    if filter_mode == "value":
        print(f"   Filter epsilon: {filter_epsilon}")
    elif filter_mode == "rollout":
        print(f"   Rollout horizon: {rollout_horizon}")
        print(f"   Velocity threshold: {rollout_velocity_threshold}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Step delay: {step_delay}s")
    if record_video:
        print(f"   Video recording: ON")
        print(f"   Video folder: {video_folder}")
        print(f"   Video FPS: {video_fps}")
        if video_camera:
            print(f"   Video camera: {video_camera}")
    print(f"\nGoal: Navigate using trained policy {'with safety filter' if filter_mode != 'none' else 'without filter'}!")
    print(f"   Safety violations will be tracked via environment's built-in cost signal\n")
    
    episode_stats = []
    
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs, info = env.reset()
            
            # If using initial states dataset, soft reset to the specified state
            if initial_states is not None:
                state_index = run_index + episode
                state = initial_states[state_index]
                
                print(f"  Soft resetting to state index {state_index}")
                
                env.soft_reset()
                
                # Use soft_reset() and set_state() as in the rollout filter
                if 'mujoco_state' in state and hasattr(env, 'task') and hasattr(env.task, 'world'):
                    # Perform soft reset (resets time limit wrapper without full reset)
                    env.soft_reset()
                    
                    # Get world and set the saved state
                    world = env.task.world
                    world.set_state(state['mujoco_state'])
                    
                    # Forward the physics to update everything
                    mujoco.mj_forward(world.model, world.data)
                    
                    # Get observation from the set state
                    obs = env._get_obs() if hasattr(env, '_get_obs') else state['observation']
                    
                    print(f"  State reset successful using World API")
                else:
                    print(f"  Warning: Could not perform soft reset (mujoco_state or World API not available)")
            
            episode_reward = 0.0
            episode_cost = 0.0  # Track safety violations via cost
            episode_steps = 0
            cost_violations = 0
            filter_interventions = 0
            
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
                
                # Track filter interventions
                if 'action_filtered' in info and info['action_filtered']:
                    filter_interventions += 1
                
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
                'filter_interventions': filter_interventions,
                'terminated': terminated,
                'truncated': truncated
            }
            episode_stats.append(final_info)
            
            print(f"      Episode {episode + 1} complete!")
            print(f"      Reward: {episode_reward:.3f}")
            print(f"      Cost: {episode_cost:.3f}")
            print(f"      Steps: {episode_steps}")
            print(f"      Cost violations: {cost_violations}")
            if filter_mode != "none":
                print(f"      Filter interventions: {filter_interventions}")
                intervention_rate = (filter_interventions / episode_steps * 100) if episode_steps > 0 else 0
                print(f"      Intervention rate: {intervention_rate:.1f}%")
            print(f"      Reason: {'Task complete/Safety violation' if terminated else 'Timeout' if truncated else 'Complete'}")
            print()
            
            # Brief pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"\nReplay interrupted by user")
    
    finally:
        env.close()
        
        # Print overall statistics
        if episode_stats:
            print(f"\nOverall Statistics ({len(episode_stats)} episodes):")
            avg_reward = sum(ep['reward'] for ep in episode_stats) / len(episode_stats)
            avg_cost = sum(ep['cost'] for ep in episode_stats) / len(episode_stats)
            avg_steps = sum(ep['steps'] for ep in episode_stats) / len(episode_stats)
            total_violations = sum(ep['cost_violations'] for ep in episode_stats)
            
            print(f"   Average reward: {avg_reward:.3f}")
            print(f"   Average cost: {avg_cost:.3f}")
            print(f"   Average steps: {avg_steps:.1f}")
            print(f"   Total cost violations: {total_violations}")
            
            if filter_mode != "none":
                total_interventions = sum(ep['filter_interventions'] for ep in episode_stats)
                total_steps = sum(ep['steps'] for ep in episode_stats)
                avg_intervention_rate = (total_interventions / total_steps * 100) if total_steps > 0 else 0
                print(f"   Total filter interventions: {total_interventions}")
                print(f"   Overall intervention rate: {avg_intervention_rate:.1f}%")
            
            # Safety rate (episodes with zero cost)
            safe_episodes = sum(1 for ep in episode_stats if ep['cost'] == 0)
            safety_rate = safe_episodes / len(episode_stats) * 100
            print(f"   Safety rate: {safety_rate:.1f}% ({safe_episodes}/{len(episode_stats)} episodes with zero cost)")



def main():
    parser = argparse.ArgumentParser(
        description="Replay a trained SAC model on safety-gymnasium with optional safety filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay without filter (default)
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip

  # Replay with safety value filter
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --filter value --safety-model path/to/safety_model.zip --epsilon 0.0

  # Replay with safety rollout filter
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --filter rollout --safety-model path/to/safety_model.zip \\
      --horizon 10 --velocity-threshold 0.1
  
  # Record videos of episodes
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --record-video --video-folder ./my_videos --video-fps 30
  
  # Record videos with different camera view (top-down)
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --record-video --video-camera fixednear
  
  # Record videos with tracking camera
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --record-video --video-camera track
  
  # Record videos with rollout filter
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --filter rollout --safety-model path/to/safety_model.zip \\
      --record-video --video-folder ./filtered_videos
  
  # Use pre-generated initial states for consistent evaluation
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --initial-states initial_states/SafetyCarCircle2-v0_n100.pkl \\
      --run-index 0 --episodes 10
  
  # Evaluate with initial states and video recording
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --initial-states initial_states/SafetyCarCircle2-v0_n100.pkl \\
      --run-index 0 --episodes 5 --record-video --video-camera fixednear
        """
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to the trained SAC model file (.zip)")
    parser.add_argument("--env", type=str, default="SafetyCarCircle2-v0",
                        choices=["SafetyCarCircle1-v0", "SafetyCarCircle2-v0", 
                                "SafetyPointCircle1-v0", "SafetyPointCircle2-v0"],
                        help="Safety-gymnasium environment ID")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--delay", "-d", type=float, default=0.02,
                        help="Delay between steps for visualization (seconds)")
    
    # Filter options
    parser.add_argument("--filter", type=str, default="none",
                        choices=["none", "value", "rollout"],
                        help="Safety filter mode: 'none' (no filter), 'value' (Q-value based), 'rollout' (simulation based)")
    parser.add_argument("--safety-model", type=str, default=None,
                        help="Path to trained SafetySAC model (required for value/rollout filters)")
    
    # Value filter options
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Safety margin threshold for value filter (default: 0.0)")
    
    # Rollout filter options
    parser.add_argument("--horizon", type=int, default=10,
                        help="Rollout horizon for rollout filter (default: 10)")
    parser.add_argument("--velocity-threshold", type=float, default=0.1,
                        help="Velocity threshold for rollout filter (default: 0.1)")
    
    # Video recording options
    parser.add_argument("--record-video", action="store_true",
                        help="Record videos of episodes")
    parser.add_argument("--video-folder", type=str, default="./videos",
                        help="Directory to save video recordings (default: ./videos)")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="Frame rate for video recordings (default: 30)")
    parser.add_argument("--video-camera", type=str, default="fixednear",
                        choices=[None, "vision", "track", "fixednear", "fixedfar", "human"],
                        help="Camera to use for video recording (default: environment default, usually 'vision')")
    
    # Initial states options
    parser.add_argument("--initial-states", type=str, default=None,
                        help="Path to pickle file with initial states for consistent evaluation")
    parser.add_argument("--run-index", type=int, default=0,
                        help="Starting index in initial states dataset (default: 0)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Make sure you have trained a SAC model first")
        return
    
    # Validate filter configuration
    if args.filter in ["value", "rollout"] and args.safety_model is None:
        print(f"Error: --safety-model is required when using --filter {args.filter}")
        return
    
    replay_sac_model(
        model_path=args.model,
        env_id=args.env,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        step_delay=args.delay,
        filter_mode=args.filter,
        safety_model_path=args.safety_model,
        filter_epsilon=args.epsilon,
        rollout_horizon=args.horizon,
        rollout_velocity_threshold=args.velocity_threshold,
        record_video=args.record_video,
        video_folder=args.video_folder,
        video_fps=args.video_fps,
        video_camera=args.video_camera,
        initial_states_file=args.initial_states,
        run_index=args.run_index
    )


if __name__ == "__main__":
    main()