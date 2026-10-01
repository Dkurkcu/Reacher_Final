import numpy as np
import os
from stable_baselines3 import PPO
from env.reacher_latest import ReacherV3Env 
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack 

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
# Options: 
# "black_nobox" (Model: No penalties/No pullback, Env: No Box)       -> use black_model.zip
# "blue_nobox"  (Model: Pullback architecture,    Env: No Box)       -> use blue_model.zip
# "black_light" (Model: No penalties/No pullback, Env: Light Box)    -> use boxedblack_model.zip
# "green_light" (Model: Penalties active/No pull, Env: Light Box)    -> use green_model.zip
# "black_heavy" (Model: No penalties/No pullback, Env: Heavy Box)    -> use boxedblack_model.zip
# "blue_heavy"  (Model: Pullback architecture,    Env: Heavy Box)    -> use blue_model.zip

SCENARIO_NAME = "blue_heavy"  

# Point this to the exact brain you want to test for this specific scenario
MODEL_PATH = "blue_model.zip" 

# The specific seeds you want to extract data for
SEEDS = [222,228,1296,3291,5597]
# ==========================================

def record_data():
    os.makedirs("plot_data", exist_ok=True)

    # 1. Setup Environment
    env = DummyVecEnv([lambda: ReacherV3Env()])
    env = VecFrameStack(env, n_stack=4)
    model = PPO.load(MODEL_PATH)

    # 2. Internal Mapper for File Saving
    valid_scenarios = {
        "black_nobox": ("black", "nobox"),
        "blue_nobox":  ("blue", "nobox"),
        "black_light": ("black", "light"),
        "green_light": ("green", "light"),
        "blue_light":  ("blue", "light"),  # Added just in case
        "black_heavy": ("black", "heavy"),
        "green_heavy": ("green", "heavy"), # Added just in case
        "blue_heavy":  ("blue", "heavy")
    }

    if SCENARIO_NAME not in valid_scenarios:
        print("[Error] Unknown SCENARIO_NAME. Check your configuration.")
        return
        
    color, env_type = valid_scenarios[SCENARIO_NAME]
    print(f"--- Recording {color.upper()} line data in {env_type.upper()} environment ---")

    for seed in SEEDS:
        print(f"Running Seed {seed}...")
        
        obs = env.reset()
        env.envs[0].np_random = np.random.default_rng(seed) 
        obs = env.reset() 
        
        dist_log, elec_log, mech_log, kin_log = [], [], [], []
        
        for step in range(650):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            info = infos[0] 
            
            # --- Extract Telemetry ---
            current_dist = info.get("distance", 0.0)
            current_elec = float(np.sum(np.square(action[0])))
            current_mech = abs(info.get("energy_inc", 0.0))
            
            qvel_array = info.get("qvel", np.zeros(2))
            current_kin = float(np.sum(np.square(qvel_array)))
            
            dist_log.append(current_dist)
            elec_log.append(current_elec)
            mech_log.append(current_mech)
            kin_log.append(current_kin)
            
            # Pad arrays if episode ends early to maintain 500 length
            if dones[0]:
                remaining = 650 - (step + 1)
                dist_log.extend([current_dist] * remaining)
                elec_log.extend([0.0] * remaining)
                mech_log.extend([0.0] * remaining)
                kin_log.extend([0.0] * remaining)
                break

        # Save using the mapped prefix (nobox, light, or heavy)
        np.save(f"plot_data/{color}_dist_{env_type}_seed_{seed}.npy", np.array(dist_log))
        np.save(f"plot_data/{color}_elec_{env_type}_seed_{seed}.npy", np.array(elec_log))
        np.save(f"plot_data/{color}_mech_{env_type}_seed_{seed}.npy", np.array(mech_log))
        np.save(f"plot_data/{color}_kin_{env_type}_seed_{seed}.npy", np.array(kin_log))
        
        print(f"  -> Saved {color}_[metric]_{env_type} data.")

if __name__ == "__main__":
    record_data()