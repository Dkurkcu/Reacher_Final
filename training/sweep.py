# sweep.py
import os
import yaml
import numpy as np
import wandb
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium.wrappers import TimeLimit

# NOTE: We removed CleanResetWrapper. 
# DummyVecEnv handles the (obs, info) tuple correctly by default.

def train():
    wandb.init()
    config = wandb.config

    # --- Metrics Setup ---
    wandb.define_metric("global/step")
    wandb.define_metric("*", step_metric="global/step", step_sync=True)
    wandb.define_metric("ep/step", hidden=True)
    wandb.define_metric("ep/*", step_metric="ep/step") 
    
    from env.reacher_latest import ReacherV3Env 

    log_dir = f"./sweep_logs/{wandb.run.id}"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(seed=None):
        def _init():
            env = ReacherV3Env("reacher_v3.xml")
            
            if seed is not None:
                try:
                    env.action_space.seed(int(seed))
                    env.reset(seed=int(seed))
                except Exception:
                    pass

            env = TimeLimit(env, max_episode_steps=500)
            
            env = Monitor(
                env,
                filename=os.path.join(log_dir, "monitor.csv"),
                info_keywords=(
                    "episode_energy", 
                    "success", 
                    "distance",
                    "episode_stall_energy",
                    "stall_energy_limit_reached",
                ),
            )
            
            # NO WRAPPER HERE - Standard Gymnasium API is (obs, info)
            return env
        return _init

    # ---------- Logging Callback ----------
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
                    ep_stall_energy = float(info.get("episode_stall_energy", 0.0))
                    
                    wandb.log(
                        {
                            "global/step": self.num_timesteps, 
                            "metrics/success_rate": sr_window,
                            "metrics/episode_reward": ep_r,
                            "metrics/episode_energy": ep_energy,
                            "metrics/episode_stall_energy_total": ep_stall_energy, 
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

    seed_value = int(config.seed)
    np.random.seed(seed_value)

    # ------------------- SINGLE PROCESS SETUP -------------------
    N_ENVS = 4
    N_STACK = 4
    
    # DummyVecEnv handles the Gymnasium (obs, info) tuple natively
    env_fns = [make_env(seed=seed_value + i) for i in range(N_ENVS)]
    venv = DummyVecEnv(env_fns) 

    venv = VecFrameStack(venv, n_stack=N_STACK) 
    venv.seed(seed_value)
    # ------------------------------------------------------------

    learning_rate = float(config.learning_rate)
    ent_coef = float(config.ent_coef)
    batch_size = int(config.batch_size)
    clip_range = float(config.clip_range)

    policy_kwargs = dict(net_arch=[256, 256, 256])

    model = PPO(
        "MlpPolicy",
        env=venv, 
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        clip_range=clip_range,
        seed=seed_value,
        verbose=1,
    )

    # Eval Env (Single Process)
    eval_seed = seed_value + 10_000
    N_EVAL_ENVS = 2 
    eval_env_fns = [make_env(seed=eval_seed + i) for i in range(N_EVAL_ENVS)]
    eval_env_base = DummyVecEnv(eval_env_fns)
    eval_env = VecFrameStack(eval_env_base, n_stack=N_STACK) 
    eval_env.seed(eval_seed)

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

    model.save("green_model.zip")
    wandb.finish()
    venv.close()
    eval_env.close()

if __name__ == "__main__":
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="energy_tracking_with_penalties")
    wandb.agent(sweep_id, function=train)