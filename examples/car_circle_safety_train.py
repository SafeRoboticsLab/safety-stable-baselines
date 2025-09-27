# train_safety_sac_circle_wandb.py
import os
import sys
import time
from datetime import datetime
import wandb

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback  # syncs TB logs + optional model checkpoints

from safety_gymnasium.safety_envs.safety_circle_margin import make_env

# so "from safety_sb3 import SafetySAC" works when running from /examples
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety_sb3 import SafetySAC


if __name__ == "__main__":
    # ---------- configuration ----------
    # Experiment identifier - add suffix/prefix to distinguish experiment sets
    # Examples: "_test1", "_ablation", "_final", "_geometric", "_v2", etc.
    EXP_SUFFIX = ""  # Set to "" for no suffix, or e.g. "_geometric" for identification
    
    # ---------- paths ----------
    base_run_name = "SafetySAC_CarCircle2"
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
            "algo": "SafetySAC",
            "env_id": "SafetyCarCircle2-v0",
            "safety_clearance": 0.0,
            "exp_suffix": EXP_SUFFIX,
            "total_timesteps": 500_000,
            "lr": 1e-5,
            "buffer_size": 200_000,
            "batch_size": 256,
            "gamma": 0.995,
            "tau": 0.01,
        },
        sync_tensorboard=True,         # auto-sync SB3 TensorBoard logs
        save_code=True,
    )

    # ---------- env ----------
    # NOTE: SB3 auto-wraps with Monitor, but we do it explicitly so episodic stats are guaranteed.
    env = make_env(agent="Car", level=2, render_mode=None, safety_clearance=0.0)
    env = Monitor(env)  # ensures episodic reward/length are logged

    # Separate eval env (no render)
    eval_env = make_env(agent="Car", level=2, render_mode=None, safety_clearance=0.0)
    eval_env = Monitor(eval_env)

    # ---------- model ----------
    model = SafetySAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-5,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.01,
        gamma=0.995,                 # safety discount
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        seed=0,
        device="auto",
        verbose=1,
        tensorboard_log=logs_dir,    # <-- enable TB so W&B can sync it
    )

    # ---------- callbacks ----------
    # 1) Save "best model" based on periodic eval
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=10_000,           # eval every N steps
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    # 2) Periodic checkpoints (latest snapshots during training)
    ckpt_cb = CheckpointCallback(
        save_freq=10_000,           # save every N steps
        save_path=ckpt_dir,
        name_prefix="safety_sac_car_circle2",
        save_replay_buffer=True,    # useful for resuming
        save_vecnormalize=False,
    )

    # 3) W&B callback (also supports saving models to W&B Artifacts)
    wb_cb = WandbCallback(
        gradient_save_freq=0,                     # set >0 to log gradients every k updates
        model_save_path=os.path.join(ckpt_dir, "wandb"),  # optional extra checkpoints via W&B
        model_save_freq=0,                        # set >0 to also save via W&B every k steps
        verbose=2,
    )

    callbacks = CallbackList([eval_cb, ckpt_cb, wb_cb])

    # ---------- train ----------
    model.learn(
        total_timesteps=500_000,
        callback=callbacks,
        tb_log_name=run_name,       # TB run group name (appears in W&B)
        log_interval=10,            # print/log every 10 train calls
    )

    # ---------- final save ----------
    final_path = os.path.join(final_dir, "car_circle2")
    model.save(final_path)
    print(f"Training complete! Saved final SafetySAC model to {final_path}.zip")

    # ---------- tidy up ----------
    env.close()
    eval_env.close()
    wandb_run.finish()
