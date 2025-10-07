import os
import sys
from datetime import datetime
import argparse
import wandb
import safety_gymnasium
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
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
    # ---------- argument parsing ----------
    parser = argparse.ArgumentParser(description="Train SAC on CarCircle2 (naive, no safety filter)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--exp-suffix", type=str, default="", help="Experiment identifier suffix for distinguishing experiment sets")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total training timesteps")
    args = parser.parse_args()

    SEED = args.seed
    EXP_SUFFIX = args.exp_suffix
    TOTAL_TIMESTEPS = args.total_timesteps

    # ---------- paths ----------
    base_run_name = "SAC_CarCircle2"
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
            "algo": "SAC",
            "env_id": "SafetyCarCircle2-v0",
            "exp_suffix": EXP_SUFFIX,
            "total_timesteps": TOTAL_TIMESTEPS,
            "lr": 3e-4,
            "buffer_size": 100_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.01,
            "seed": SEED,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # ---------- env ----------
    env = safety_gymnasium.make("SafetyCarCircle2-v0")
    env = TerminateOnCollisionWrapper(env)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    env = Monitor(env)

    eval_env = safety_gymnasium.make("SafetyCarCircle2-v0")
    eval_env = TerminateOnCollisionWrapper(eval_env)
    eval_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(eval_env)
    eval_env = Monitor(eval_env)

    # ---------- model ----------
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        seed=SEED,
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
        save_freq=10_000,
        save_path=ckpt_dir,
        name_prefix="sac_car_circle2",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    wb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(ckpt_dir, "wandb"),
        model_save_freq=0,
        verbose=2,
    )

    safety_cb = SafetyLoggingCallback(verbose=1)

    callbacks = CallbackList([eval_cb, ckpt_cb, wb_cb, safety_cb])

    # ---------- train ----------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name=run_name,
        log_interval=10,
    )

    # ---------- final save ----------
    final_path = os.path.join(final_dir, "car_circle2_sac")
    model.save(final_path)
    print(f"Training complete! Saved final SAC model to {final_path}.zip")

    # ---------- tidy up ----------
    env.close()
    eval_env.close()
    wandb_run.finish()
