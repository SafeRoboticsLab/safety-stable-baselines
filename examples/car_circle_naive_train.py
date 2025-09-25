# train_sac_circle_wandb.py - Simple SAC training on original safety-gymnasium
import os
import sys
import datetime
import wandb
import safety_gymnasium

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# so imports work when running from /examples
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # ---------- paths ----------
    run_name = f"SafetySAC_CarCircle2_{datetime.now().strftime('%Y%m%d_%H%M')}"
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
            "total_timesteps": 100_000,
            "lr": 3e-4,
            "buffer_size": 100_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.01,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # ---------- env ----------
    # Use original safety-gymnasium environment directly
    env = safety_gymnasium.make("SafetyCarCircle2-v0")
    env = TerminateOnCollisionWrapper(env)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    env = Monitor(env)

    # Separate eval env
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
        learning_starts=5_000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
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

    callbacks = CallbackList([eval_cb, ckpt_cb, wb_cb])

    # ---------- train ----------
    model.learn(
        total_timesteps=100_000,
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
