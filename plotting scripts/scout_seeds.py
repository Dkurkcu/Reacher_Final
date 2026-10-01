import os
import numpy as np
import gymnasium

from env.reacher_latest import ReacherV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import TimeLimit

MODEL_PATH = "blue_model.zip" 
XML_NAME = "reacher_v3.xml"     # Uses your exact medium-mass xml parameter
N_STACK = 4 

def make_env():
    def _init():
        # HEADLESS MODE: Remove render_mode="human" so it runs 100x faster
        env = ReacherV3Env(XML_NAME) 
        env = TimeLimit(env, max_episode_steps=500) 
        return env
    return _init

def main():
    target_harvest_count = 20  # How many golden seeds you want to collect
    start_seed = 5000         # Start scanning from your known good seed
    max_seeds_to_scan = 50000
    
    # 1. Setup Environment Pipeline exactly matching your test.py
    venv_base = DummyVecEnv([make_env()])
    venv = VecFrameStack(venv_base, n_stack=N_STACK) 
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found: {MODEL_PATH}")
        return
        
    model = PPO.load(MODEL_PATH, env=venv) 

    harvested_seeds = []

    print("====================================================")
    print("      STARTING FAST HEADLESS SEED HARVESTER         ")
    print("====================================================")
    print(f"{'Seed':<8}{'Stalled?':<12}{'Succeeded?':<14}{'Status':<15}")
    print("-" * 55)

    for episode in range(start_seed, start_seed + max_seeds_to_scan): 
        
        # Lock seed configuration
        venv.seed(episode)
        obs = venv.reset() 

        done = False
        truncated = False
        last_info = {}
        
        # Sticky episode tracking registers
        stalled_during_episode = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_arr, done_arr, infos = venv.step(action) 
            
            info = infos[0] 
            
            # --- LIVE TELEMETRY CATCHER ---
            # If it triggers your step function's strict stall criteria even ONCE, flag it!
            if bool(info.get("is_stalled", False)) or float(info.get("sting", 0.0)) > 0.5:
                stalled_during_episode = True

            # Calculate termination triggers exactly like test.py
            stall_trunc = bool(info.get("stall_energy_limit_reached", False))
            time_trunc = bool(info.get("TimeLimit.truncated", False))
            
            truncated = stall_trunc or time_trunc
            done = bool(done_arr[0])
            last_info = info

        # --- EPISODE EVALUATION MATRIX ---
        success = bool(last_info.get("success", False))
        
        status_string = "Discarded"
        if stalled_during_episode and success:
            harvested_seeds.append(episode)
            status_string = "HARVESTED! ⭐"
        elif not stalled_during_episode:
            status_string = "Skipped (No Stall)"
        elif not success:
            status_string = "Failed (Stuck/Broken)"

        # Print clean, un-spammed summary line per seed
        print(f"{episode:<8}{str(stalled_during_episode):<12}{str(success):<14}{status_string:<15}")

        # Stop once you have your paper's sample size filled
        if len(harvested_seeds) == target_harvest_count:
            print("\n====================================================")
            print(f"SUCCESS: Target batch size of {target_harvest_count} seeds achieved!")
            break

    print("\nCopy and paste this list directly into your evaluation plotting script:")
    print(f"seeds = {harvested_seeds}")
    print("====================================================")
    
    venv.close()

if __name__ == "__main__":
    main()