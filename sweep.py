# sweep.py
import os
import yaml
import numpy as np
import wandb
import multiprocessing as mp 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium.wrappers import TimeLimit
# ADD THIS IMPORT
from stable_baselines3.common.vec_env import VecFrameStack


# Set the start method for multiprocessing 
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def train():
    wandb.init()
    config = wandb.config

    # --- W&B Step Synchronization and Metric Definition ---
    wandb.define_metric("global/step")
    wandb.define_metric("*", step_metric="global/step", step_sync=True)
    
    wandb.define_metric("ep/step", hidden=True)
    wandb.define_metric("ep/*", step_metric="ep/step") 
    # ------------------------------------------------------

    from reacher_latest import ReacherV3Env 

    log_dir = f"./sweep_logs/{wandb.run.id}"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(seed=None):
        def _init():
            env = ReacherV3Env("reacher_v3.xml")

            if seed is not None:
                try:
                    env.action_space.seed(int(seed))
                except Exception:
                    pass
                try:
                    env.reset(seed=int(seed))
                except Exception:
                    pass

            # Updated max episode steps to 800
            env = TimeLimit(env, max_episode_steps=400)

            env = Monitor(
                env,
                filename=os.path.join(log_dir, "monitor.csv"),
                info_keywords=(
                    "episode_energy", "success", "distance",
                    "episode_contact_energy", "contact_energy_limit_reached",
                   # "collision_penalty", 
                ),
            )
            return env
        return _init

    # ---------- Minimal logging: ReacherLoggingCallback (Unchanged) ----------
    class ReacherLoggingCallback(BaseCallback):
        def __init__(self, window=100, verbose=0):
            super().__init__(verbose)
            self.window = int(window)
            self._cur_success = None
            self.ep_success = []
            self._step_in_ep = None
            self._first_success_step = None
            self._episode_index = 0

        def _on_training_start(self) -> None:
            n_envs = getattr(self.training_env, "num_envs", 1)
            self._cur_success = [False] * n_envs
            self._step_in_ep = [0] * n_envs
            self._first_success_step = [None] * n_envs

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for i, info in enumerate(infos):
                self._step_in_ep[i] += 1
                if info.get("success", False) and self._first_success_step[i] is None:
                    self._first_success_step[i] = self._step_in_ep[i]
                    self._cur_success[i] = True

                if "episode" in info and "r" in info["episode"]:
                    ep_r = float(info["episode"]["r"])
                    ep_success = 1 if self._cur_success[i] else 0
                    self._cur_success[i] = False

                    self.ep_success.append(ep_success)
                    sr_window = float(np.mean(self.ep_success[-self.window:])) if self.ep_success else 0.0

                    ep_energy = float(info.get("episode_energy", 0.0))
                    ep_contact_energy = float(info.get("episode_contact_energy", 0.0))
                    
                    wandb.log(
                        {
                            "global/step": self.num_timesteps, 
                            "metrics/success_rate": sr_window,
                            "metrics/episode_reward": ep_r,
                            "metrics/episode_energy": ep_energy,
                            "metrics/episode_contact_energy_total": ep_contact_energy, 
                        }
                    )

                    final_distance = float(np.asarray(info.get("distance", np.nan)))
                    steps_to_success = (
                        float(self._first_success_step[i])
                        if self._first_success_step[i] is not None else float("nan")
                    )
                    self._episode_index += 1
                    wandb.log(
                        {
                            "ep/step": self._episode_index, 
                            "ep/final_distance": final_distance,
                            "ep/steps_to_success": steps_to_success,
                            "ep/success": ep_success,
                        }
                    )

                    self._step_in_ep[i] = 0
                    self._first_success_step[i] = None

            return True

    # Seed from sweep config
    seed_value = int(config.seed)
    np.random.seed(seed_value)

    # ------------------- MULTICORE & FRAMESTACK SETUP -------------------
    N_ENVS = 4    # Number of parallel environments
    N_STACK = 4   # Number of frames to stack (K=4)
    
    # 1. Create base vectorized environments (multiprocess)
    env_fns = [make_env(seed=seed_value + i) for i in range(N_ENVS)]
    venv_base = SubprocVecEnv(env_fns) 

    # 2. Add memory via FrameStack
    venv = VecFrameStack(venv_base, n_stack=N_STACK) 
    
    try:
        venv.seed(seed_value)
    except Exception:
        pass
    # --------------------------------------------------------------------

    # Hyperparams from sweep (Unchanged)
    learning_rate = float(config.learning_rate)
    ent_coef = float(config.ent_coef)
    batch_size = int(config.batch_size)
    clip_range = float(config.clip_range)

    policy_kwargs = dict(net_arch=[256, 256, 256])

    model = PPO(
        "MlpPolicy",
        env=venv, # Uses the FrameStacked environment
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        clip_range=clip_range,
        seed=seed_value,
        verbose=1,
    )

    # Apply stacking to the evaluation environment as well
    eval_seed = seed_value + 10_000
    N_EVAL_ENVS = 2 
    eval_env_fns = [make_env(seed=eval_seed + i) for i in range(N_EVAL_ENVS)]
    eval_env_base = SubprocVecEnv(eval_env_fns)
    eval_env = VecFrameStack(eval_env_base, n_stack=N_STACK) # FrameStack here too

    try:
        eval_env.seed(eval_seed)
    except Exception:
        pass

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=5,
        verbose=1,
    )

    model.learn(
        total_timesteps=4_000_000,
        callback=[ReacherLoggingCallback(window=100), eval_callback],
    )

    model.save("fixed_arm_directsweep.zip")
    wandb.finish()
    venv.close()
    eval_env_base.close() # Close the base envs
    eval_env.close()

if __name__ == "__main__":
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project="energy_tracking_with_penalties")
    wandb.agent(sweep_id, function=train)