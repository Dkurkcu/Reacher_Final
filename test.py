import os
import time
import numpy as np

# Assuming you saved your environment as reacher_v3.py
from reacher_latest import ReacherV3Env # your env module/class
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit


MODEL_PATH = "fixed_arm_directsweep.zip" # adjust if needed
XML_NAME = "reacher_v3.xml"# adjust if needed


def make_env():
    def _init():
        env = ReacherV3Env(XML_NAME, render_mode="human") 
        env = TimeLimit(env, max_episode_steps=800)
        return env
    return _init


def unwrap_env(venv):
    """Unwraps the environment from VecEnv, TimeLimit, etc., to get the base env."""
    env = venv.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def main():
    # Build single-env VecEnv with human render
    venv = DummyVecEnv([make_env()])
    base_env = unwrap_env(venv)

    # Optional: quick wiring print (using actual attribute names from ReacherV3Env)
    try:
        print("=== Env wiring ===")
        print(f"Box geom id:          {getattr(base_env, 'box_geom_id', None)}")
        print(f"Box body id:          {getattr(base_env, 'box_body_id', None)}")
        arm_gids = sorted(list(getattr(base_env, 'arm_geom_ids', []))) if getattr(base_env, 'arm_geom_ids', None) else []
        print(f"Arm geom ids:         {arm_gids}")
        print("===================\n")
    except Exception as e:
        print(f"[Warn] Could not print env wiring: {e}")

    # Load trained policy (or exit gracefully if missing)
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found: {MODEL_PATH}")
        print("Train or place your PPO zip here, or change MODEL_PATH.")
        venv.close()
        return
    model = PPO.load(MODEL_PATH, env=venv)

    # Reset
    # FIX APPLIED HERE: Only unpack the observation
    obs = venv.reset() 

    for episode in range(200):
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        last_info = {}
        any_contact = False
        in_contact_prev = False
        exceeded_announced = False

        print(f"\n--- EPISODE {episode+1:02d} ---")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            
            # FIX APPLIED HERE: Unpack 4 values (obs, reward, done, info)
            obs, reward_arr, done_arr, infos = venv.step(action) 
            
            # Render viewer
            venv.render()
            time.sleep(0.01)

            # --- SB3 VecEnv unpack and bool conversion ---
            
            # Get the single environment's info dictionary
            info = infos[0]

            # 1. Unpack old 'done' flag and rewards
            old_done = bool(done_arr[0]) 
            reward = float(reward_arr[0])

            # 2. Derive modern flags from old 'done' and 'info'
            # Note: The custom safety check is a TRUNCATION signal.
            safety_trunc = bool(info.get("contact_energy_limit_reached", False))
            time_trunc = bool(info.get("TimeLimit.truncated", False)) 
            
            # If done is True, it means either terminated (success) or truncated (time limit/safety) occurred.
            # We treat everything that isn't safety_trunc as 'terminated' for simplicity in old logic.
            # The official Gymnasium split is: done = terminated OR truncated
            # If the episode ended (old_done=True) AND it was due to safety or time limit, it's truncated.
            # Otherwise, it's terminated (e.g., success or environment specific termination).
            
            truncated = safety_trunc or time_trunc
            terminated = old_done and not truncated 
            
            # For the loop, we use the original logic (done OR truncated)
            done = terminated or truncated 

            ep_reward += reward
            steps += 1
            last_info = info

            # --- telemetry pulled from env.info ---
            contact = bool(info.get("contact", False)) 
            e_push_step  = float(info.get("contact_energy_inc", 0.0))
            e_push_ep = float(info.get("episode_contact_energy", 0.0))
            budget= float(info.get("contact_energy_limit", np.nan)) 
            budget_exceeded = safety_trunc # safety_trunc and budget_exceeded are the same

            # Contact state transitions + per-step readout while in contact
            if contact and not in_contact_prev:
                msg = f"[CONTACT-START] step={steps}"
                if np.isfinite(budget):
                    msg += f"  budget={budget:.3f} J"
                print(msg)
            
            if contact:
                any_contact = True
                if np.isfinite(budget):
                    if e_push_step > 1e-6:
                        print(f"[CONTACT] step={steps:04d}  +ΔE_contact={e_push_step:.4f} J   "
                              f"E_contact_ep={e_push_ep:.4f} / {budget:.4f} J")
                else:
                    print(f"[CONTACT] step={steps:04d}  +ΔE_contact={e_push_step:.4f} J   "
                          f"E_contact_ep={e_push_ep:.4f} J")

            if budget_exceeded and not exceeded_announced:
                if np.isfinite(budget):
                    print(f"[SAFETY] Contact push-energy EXCEEDED at step {steps}: "
                          f"{e_push_ep:.4f} J > {budget:.4f} J")
                else:
                    print(f"[SAFETY] Contact push-energy EXCEEDED at step {steps}: {e_push_ep:.4f} J")
                exceeded_announced = True
                # The episode is now truncated, which will be caught by the while loop condition.

            if in_contact_prev and not contact:
                print(f"[CONTACT-END] step={steps}  E_contact_ep={e_push_ep:.4f} J")

            in_contact_prev = contact
        
        # --- Summarize Episode ---
        dist = float(np.asarray(last_info.get("distance", np.nan)))
        success = bool(last_info.get("success", False))
        safety_trunc = bool(last_info.get("contact_energy_limit_reached", False))
        time_trunc = bool(last_info.get("TimeLimit.truncated", False)) 

        if success:
            tag = "SUCCESS!"
        elif safety_trunc:
            tag = "SAFETY TRUNC."
        elif time_trunc:
            tag = "TIME-LIMIT TRUNC."
        elif done: # Must be terminated by old logic, but not success
            tag = "TERMINATED."
        else:
            tag = "DONE."

        print(
            f"EPISODE {episode+1:02d}: {tag}  "
            f"Distance(last)={dist:.4f}  Return={ep_reward:.2f}  Steps={steps}  "
            f"AnyContact={any_contact}  "
            f"E_contact_ep={float(last_info.get('episode_contact_energy', np.nan)):.4f} J"
        )

        # Reset for next episode
        obs = venv.reset() 
        time.sleep(0.4)


if __name__ == "__main__":
    main()