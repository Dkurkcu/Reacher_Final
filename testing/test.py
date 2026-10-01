import os
import time
import numpy as np
import gymnasium

# Assuming you saved your environment as reacher_latest.py
from env.reacher_latest import ReacherV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import VecFrameStack

MODEL_PATH = "black_model.zip"   
XML_NAME = "reacher_v3.xml"
N_STACK = 4 

def make_env():
    def _init():
        env = ReacherV3Env(XML_NAME, render_mode="human") 
        env = TimeLimit(env, max_episode_steps=500) 
        return env
    return _init

def main():
    # --- YOUR HARVESTED SEEDS LIST ---
    # Replace these placeholder numbers with the exact 20 seeds you found 
    # using your harvester script
    seeds_to_observe =  [ 222,228,1296,3291,5597]#222,228,1296,3291,5597
    # 1. Setup Environment
    venv_base = DummyVecEnv([make_env()])
    venv = VecFrameStack(venv_base, n_stack=N_STACK) 
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found: {MODEL_PATH}")
        return
    
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=venv) 

    # 3. THE TARGETED OBSERVATION LOOP
    # We step precisely through your list of chosen seeds back-to-back
    for idx, episode in enumerate(seeds_to_observe): 
        
        # Lock the seed for the Vectorized Environment
        venv.seed(episode)
        
        # Reset applies the seed and spawns the specific layout
        obs = venv.reset() 

        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        last_info = {}
        
        # Trackers for printing
        was_stalled_prev = False
        
        print(f"\n=========================================")
        print(f" PROGRESS: [{idx + 1}/{len(seeds_to_observe)}] ")
        print(f" VISUALIZING TARGET SEED: {episode}      ")
        print(f"=========================================")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward_arr, done_arr, infos = venv.step(action) 
            
            # Render
            venv.render()
            time.sleep(0.02) # Slow down slightly to watch

            # Extract Info
            info = infos[0] # Single env
            reward = float(reward_arr[0])
            
            # Check for flags
            stall_trunc = bool(info.get("stall_energy_limit_reached", False))
            time_trunc = bool(info.get("TimeLimit.truncated", False))
            
            truncated = stall_trunc or time_trunc
            done = bool(done_arr[0])
            
            ep_reward += reward
            steps += 1
            last_info = info

            # --- LIVE TELEMETRY ---
            is_stalled = bool(info.get("is_stalled", False))
            stall_energy = float(info.get("episode_stall_energy", 0.0))
            sting_val = float(info.get("sting", 0.0))
            
            # Print only when stall status changes to avoid spam
            if is_stalled and not was_stalled_prev:
                print(f"[STALL START] Step {steps} | Energy: {stall_energy:.2f} | Sting: {sting_val:.2f}")
            elif not is_stalled and was_stalled_prev:
                print(f"[STALL END]   Step {steps} | Energy: {stall_energy:.2f}")
                
            was_stalled_prev = is_stalled

            if stall_trunc:
                print(f"[!!!] SAFETY STOP: Max Stall Energy Exceeded ({stall_energy:.2f})")

        # Episode Summary
        dist = float(last_info.get("distance", 9.99))
        success = bool(last_info.get("success", False))
        
        result_tag = "FAIL"
        if success: result_tag = "SUCCESS"
        elif stall_trunc: result_tag = "BROKEN (Stall Limit)"
        
        print(f"=== SEED {episode} END: {result_tag} | Steps: {steps} | Reward: {ep_reward:.1f} | Final Dist: {dist:.3f} ===")
        time.sleep(1.0) # Pause for 1 second between seeds to let you view the final frame

    venv.close()
    print("\nFinished evaluating all target seeds.")

if __name__ == "__main__":
    main()