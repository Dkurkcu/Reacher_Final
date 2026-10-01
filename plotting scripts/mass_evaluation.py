import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium

from env.reacher_latest import ReacherV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import TimeLimit

# --- CONFIGURATION ---
BLUE_MODEL_PATH = "blue_model.zip"
GREEN_MODEL_PATH = "green_model.zip"  # Make sure your baseline model is named this
N_STACK = 4

# Your 20 working evaluation seeds
HARVESTED_SEEDS = [1219 ,1296,  1567 ,2186 ,2279, 
                   2705, 3198,  3271,  3291,  3311, 3736,  3885, 3938,  4012,  4124,
  4247 ,4357, 4688, 4721 , 4725 ]

MASS_CONFIGS = {
    "Light (0.1)": "reacher_light_mass.xml",
    "Medium (0.5)": "reacher_v3.xml",
    "Heavy (2.5)": "reacher_heavy_mass.xml"
}

def make_env(xml_name):
    def _init():
        env = ReacherV3Env(xml_name) 
        env = TimeLimit(env, max_episode_steps=500) 
        return env
    return _init

def evaluate_model_robustness(model_path, xml_name):
    """Evaluates a given model baseline against all 20 seeds under a specific XML mass."""
    venv_base = DummyVecEnv([make_env(xml_name)])
    venv = VecFrameStack(venv_base, n_stack=N_STACK)
    
    model = PPO.load(model_path, env=venv)
    success_count = 0
    
    for episode in HARVESTED_SEEDS:
        venv.seed(episode)
        obs = venv.reset()
        
        done = False
        truncated = False
        last_info = {}
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, infos = venv.step(action)
            
            info = infos[0]
            stall_trunc = bool(info.get("stall_energy_limit_reached", False))
            time_trunc = bool(info.get("TimeLimit.truncated", False))
            
            truncated = stall_trunc or time_trunc
            done = bool(done_arr[0])
            last_info = info
            
        if bool(last_info.get("success", False)):
            success_count += 1
            
    venv.close()
    return (success_count / len(HARVESTED_SEEDS)) * 100.0

def main():
    if not os.path.exists(BLUE_MODEL_PATH) or not os.path.exists(GREEN_MODEL_PATH):
        print("[Error] Missing one of your model zip files! Ensure both blue_model.zip and green_model.zip exist.")
        return

    blue_rates = []
    green_rates = []
    labels = list(MASS_CONFIGS.keys())
    
    # 1. Evaluate Blue Line (With Pullback)
    print("Evaluating Blue Line (With Pullback Algorithm)...")
    for label, xml_file in MASS_CONFIGS.items():
        rate = evaluate_model_robustness(BLUE_MODEL_PATH, xml_file)
        blue_rates.append(rate)
        print(f"   {label} Success Rate: {rate:.1f}%")
        
    # 2. Evaluate Green Line (Without Pullback)
    print("\nEvaluating Green Line (Baseline PPO - No Pullback)...")
    for label, xml_file in MASS_CONFIGS.items():
        rate = evaluate_model_robustness(GREEN_MODEL_PATH, xml_file)
        green_rates.append(rate)
        print(f"   {label} Success Rate: {rate:.1f}%")

    # 3. Plotting Grouped Bar Chart
    x = np.arange(len(labels))  # Label locations
    width = 0.35                 # Width of the bars

    plt.figure(figsize=(10, 6))
    
    # Render both bars side by side using matching line-color labels
    bars_blue = plt.bar(x - width/2, blue_rates, width, label='Blue Line (With Pullback)', color='#1f77b4', edgecolor='black', zorder=3)
    bars_green = plt.bar(x + width/2, green_rates, width, label='Green Line (No Pullback)', color='#2ca02c', edgecolor='black', zorder=3)
    
    # Chart Styling Details
    plt.title("Ablation Study: Pullback Algorithm vs. Standard Baseline Across Masses", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Obstacle Mass Configuration", fontsize=12, labelpad=10)
    plt.ylabel("Success Rate (%)", fontsize=12, labelpad=10)
    plt.xticks(x, labels)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.legend(loc='upper right', fontsize=11)
    
    # Add numerical value callouts above bars
    for bar in bars_blue:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 2, f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    for bar in bars_green:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 2, f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    output_filename = "ablation_mass_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nAblation chart saved successfully to workspace as: '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    main()