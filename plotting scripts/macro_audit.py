import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

def generate_macro_energy_audit():
    # ==========================================
    # INSERT YOUR 20+ SEEDS HERE
    # ==========================================
    seeds = [222,228,2243,2186 ,2279, 2705, 3198,  3271,  3291,  3311, 3736,  3885, 3938,  4012,  4124,
  4247 ,4357, 4688, 4721 , 4725 ] # Add the rest of your 20 seeds to this list!
    
    os.makedirs("plot_data/final_figures", exist_ok=True)
    
    # Trackers for our statistics
    blue_optimal = 0    # Blue succeeded AND saved energy
    blue_expensive = 0  # Blue succeeded BUT spent more energy
    blue_failed = 0     # Blue failed to reach target
    
    # Data for the bar chart
    seed_labels = []
    energy_deltas = []
    bar_colors = []

    print("Crunching Macro Statistics...")

    for seed in seeds:
        try:
            # 1. Load the distance and electrical data for the "With Box" scenarios
            dist_box_base = np.load(f"plot_data/green_dist_seed_{seed}.npy")
            elec_box_base = np.load(f"plot_data/green_elec_seed_{seed}.npy")
            
            dist_box_our = np.load(f"plot_data/blue_dist_seed_{seed}.npy")
            elec_box_our = np.load(f"plot_data/blue_elec_seed_{seed}.npy")
            
            # 2. Calculate Final Metrics
            # Success is defined as the final frame distance being < 0.05
            blue_succeeded = dist_box_our[-1] < 0.05
            
            # Total energy is the sum of the step-wise array
            total_elec_green = np.sum(elec_box_base)
            total_elec_blue = np.sum(elec_box_our)
            
            # Delta Energy: Positive means Blue SAVED energy. Negative means Blue SPENT more.
            delta_e = total_elec_green - total_elec_blue
            
            # 3. Categorize the Results
            if not blue_succeeded:
                blue_failed += 1
                color = '#7F7F7F' # Gray for failure
                delta_e = 0 # Flatten failures on the bar chart so they don't skew the visual
            elif delta_e > 0:
                blue_optimal += 1
                color = '#1F77B4' # Dark Blue for Optimal Success
            else:
                blue_expensive += 1
                color = '#AEC7E8' # Light Blue for Expensive Success

            # Store for the bar chart
            seed_labels.append(str(seed))
            energy_deltas.append(delta_e)
            bar_colors.append(color)
            
        except FileNotFoundError:
            print(f"  -> Skipping Seed {seed}: Missing .npy files. Did you record it?")
            continue

    # ==========================================
    # BUILD THE STATISTICAL FIGURE
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Macro-Level Hardware Efficiency Audit (20 Randomized Obstacle Environments)", 
                 fontsize=18, fontweight='bold', y=1.02)

    # ------------------------------------------
    # Subplot 1: The Percentage Breakdown (Pie Chart)
    # ------------------------------------------
    labels = [
        'Optimal Success\n(Detour Saved Energy)', 
        'Expensive Success\n(Safety Premium Paid)', 
        'Failure\n(Target Missed)'
    ]
    sizes = [blue_optimal, blue_expensive, blue_failed]
    pie_colors = ['#1F77B4', '#AEC7E8', '#7F7F7F']
    
    # Explode the optimal slice slightly for visual emphasis
    explode = (0.05, 0, 0)  

    axs[0].pie(sizes, explode=explode, labels=labels, colors=pie_colors, autopct='%1.1f%%',
               shadow=False, startangle=140, textprops={'fontsize': 12})
    axs[0].axis('equal') 
    axs[0].set_title("Overall Operational Outcomes", fontsize=15, pad=15, fontweight='bold')

    # ------------------------------------------
    # Subplot 2: The Energy Delta (Bar Chart)
    # ------------------------------------------
    x_positions = np.arange(len(seed_labels))
    bars = axs[1].bar(x_positions, energy_deltas, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    # Draw a bold line at zero
    axs[1].axhline(0, color='black', linewidth=1.5)
    
    axs[1].set_title("Net Electrical Energy Saved per Episode", fontsize=15, pad=15, fontweight='bold')
    axs[1].set_ylabel(r"Energy Delta ($\Delta E = E_{baseline} - E_{pullback}$)", fontsize=13)
    axs[1].set_xlabel("Environment Seed", fontsize=13)
    axs[1].set_xticks(x_positions)
    axs[1].set_xticklabels(seed_labels, rotation=45, ha='right')

    # Add a custom legend for the bar chart
    legend_elements = [
        Patch(facecolor='#1F77B4', edgecolor='black', label='Optimal Success (Energy Saved)'),
        Patch(facecolor='#AEC7E8', edgecolor='black', label='Expensive Success (Safety Premium Paid)'),
        Patch(facecolor='#7F7F7F', edgecolor='black', label='Task Failed')
    ]
    axs[1].legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Clean up layout and save
    plt.tight_layout()
    save_path = "plot_data/final_figures/Macro_Energy_Audit_20Seeds.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n-> Successfully generated {save_path}")

if __name__ == "__main__":
    generate_macro_energy_audit()