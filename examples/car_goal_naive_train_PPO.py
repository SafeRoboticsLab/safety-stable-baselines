import os
import sys
from datetime import datetime
import wandb
import safety_gymnasium
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# so imports work when running from /examples
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SafetyLoggingCallback(BaseCallback):
    """
    Custom callback to log safety violations to wandb and tensorboard.
    Tracks cost violations from the safety-gymnasium environment.
    """
    
    def __init__(self, verbose=0):
        super(SafetyLoggingCallback, self).__init__(verbose)
        self.episode_costs = []
        self.episode_violations = []
        self.current_episode_cost = 0
        self.current_episode_violations = 0
        
        # Track total violations from start to finish
        self.total_violations = 0
        
    def _on_step(self) -> bool:
        # Extract info from the last step
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]  # Get info from first environment
            
            # Track cost from safety-gymnasium
            if 'cost' in info:
                cost = info['cost']
                self.current_episode_cost += cost
                if cost > 0:
                    self.current_episode_violations += 1
                    self.total_violations += 1  # Increment total violations counter
        
        # Check if episode is done
        dones = self.locals.get('dones', [])
        if len(dones) > 0 and dones[0]:  # Episode finished
            # Store episode statistics
            self.episode_costs.append(self.current_episode_cost)
            self.episode_violations.append(self.current_episode_violations)
            
            # Log to tensorboard and wandb
            self.logger.record("safety/episode_cost", self.current_episode_cost)
            self.logger.record("safety/episode_violations", self.current_episode_violations)
            self.logger.record("safety/total_violations", self.total_violations)  # Log cumulative total
            
            # Log cumulative statistics
            if len(self.episode_costs) > 0:
                avg_cost = np.mean(self.episode_costs[-100:])  # Last 100 episodes
                avg_violations = np.mean(self.episode_violations[-100:])
                violation_rate = np.mean([1 if v > 0 else 0 for v in self.episode_violations[-100:]])
                
                self.logger.record("safety/avg_episode_cost_100", avg_cost)
                self.logger.record("safety/avg_episode_violations_100", avg_violations)
                self.logger.record("safety/violation_rate_100", violation_rate)
            
            # Reset for next episode
            self.current_episode_cost = 0
            self.current_episode_violations = 0
            
        return True

if __name__ == "__main__":
    # ---------- configuration ----------
    # Experiment identifier - add suffix/prefix to distinguish experiment sets
    # Examples: "_test1", "_ablation", "_final", "_baseline", "_v2", etc.
    EXP_SUFFIX = ""  # Set to "" for no suffix, or e.g. "_baseline" for identification
    
    # ---------- paths ----------
    base_run_name = "PPO_CarGoal2"
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{base_run_name}_{EXP_SUFFIX}"
    logs_dir = f"./experiments/{run_name}/logs"
    ckpt_dir = f"./experiments/{run_name}/checkpoints"
    best_dir = f"./experiments/{run_name}/best"
    final_dir = f"./experiments/{run_name}/final"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # ---------- W&B ----------
    wandb_run = wandb.init(
        project="safety_sb3",
        entity="safe-princeton",
        name=run_name,
        config={
            "algo": "PPO",
            "env_id": "SafetyCarGoal2-v0",
            "exp_suffix": EXP_SUFFIX,
            "total_timesteps": 400_000,  # PPO typically needs more timesteps
            "lr": 3e-4,
            "n_steps": 2048,  # Steps per rollout
            "batch_size": 64,  # Minibatch size
            "n_epochs": 10,    # Number of epochs per update
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,  # Entropy coefficient
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # ---------- env ----------
    # PPO works better with vectorized environments
    env = safety_gymnasium.make("SafetyCarGoal2-v0")
    env = TerminateOnCollisionWrapper(env)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # Vectorize for PPO

    # Separate eval env (also vectorized)
    eval_env = safety_gymnasium.make("SafetyCarGoal2-v0")
    eval_env = TerminateOnCollisionWrapper(eval_env)
    eval_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(eval_env)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])  # Vectorize for PPO

    # ---------- model ----------
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,        # Number of steps to run for each environment per update
        batch_size=64,       # Minibatch size
        n_epochs=10,         # Number of epoch when optimizing the surrogate loss
        gamma=0.99,
        gae_lambda=0.95,     # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range=0.2,      # Clipping parameter for PPO
        ent_coef=0.01,       # Entropy coefficient for the loss calculation
        vf_coef=0.5,         # Value function coefficient for the loss calculation
        max_grad_norm=0.5,   # Maximum value for the gradient clipping
        seed=0,
        device="auto",
        verbose=1,
        tensorboard_log=logs_dir,
    )

    # ---------- callbacks ----------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=20_000,    # Save more frequently for PPO (every ~25 updates with 2048 steps)
        save_path=ckpt_dir,
        name_prefix="ppo_car_goal2",
        save_replay_buffer=False,  # PPO doesn't use replay buffer
        save_vecnormalize=False,
    )

    wb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(ckpt_dir, "wandb"),
        model_save_freq=0,
        verbose=2,
    )

    # Safety logging callback
    safety_cb = SafetyLoggingCallback(verbose=1)

    callbacks = CallbackList([eval_cb, ckpt_cb, wb_cb, safety_cb])

    # ---------- train ----------
    model.learn(
        total_timesteps=400_000,  # PPO typically needs more timesteps than SAC
        callback=callbacks,
        tb_log_name=run_name,
        log_interval=10,
    )

    # ---------- final save ----------
    final_path = os.path.join(final_dir, "car_goal2_ppo")
    model.save(final_path)
    print(f"Training complete! Saved final PPO model to {final_path}.zip")

    # ---------- tidy up ----------
    env.close()
    eval_env.close()
    wandb_run.finish()
