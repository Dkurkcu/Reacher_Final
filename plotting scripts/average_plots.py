import numpy as np
import matplotlib.pyplot as plt
import os

def create_aggregated_4x2_grid(seeds, condition_name="Heavyweight"):
    os.makedirs("plot_data/final_figures", exist_ok=True)
    print(f"\nBuilding Aggregated 4x2 Grid for {condition_name} (Seeds: {seeds})...")

    # ==========================================
    # 1. DATA STRUCTURE SETUP
    # ==========================================
    metrics = ['dist', 'mech', 'elec', 'kin']
    
    # Left Column Data (2 Lines)
    nobox = {
        'black': {m: [] for m in metrics},
        'blue':  {m: [] for m in metrics}
    }
    
    # Right Column Data (3 Lines)
    box = {
        'black': {m: [] for m in metrics},
        'green': {m: [] for m in metrics},
        'blue':  {m: [] for m in metrics}
    }

    # ==========================================
    # 2. LOAD AND PROCESS ALL SEEDS
    # ==========================================
    valid_seeds = 0
    for seed in seeds:
        try:
            # --- Load No Box (Left) ---
            nb_d_blk = np.load(f"plot_data/black_dist_nobox_seed_{seed}.npy")
            nb_m_blk = np.load(f"plot_data/black_mech_nobox_seed_{seed}.npy")
            nb_e_blk = np.load(f"plot_data/black_elec_nobox_seed_{seed}.npy")
            nb_k_blk = np.load(f"plot_data/black_kin_nobox_seed_{seed}.npy")
            
            nb_d_blu = np.load(f"plot_data/blue_dist_nobox_seed_{seed}.npy")
            nb_m_blu = np.load(f"plot_data/blue_mech_nobox_seed_{seed}.npy")
            nb_e_blu = np.load(f"plot_data/blue_elec_nobox_seed_{seed}.npy")
            nb_k_blu = np.load(f"plot_data/blue_kin_nobox_seed_{seed}.npy")

            # --- Load With Box (Right) ---
            bx_d_blk = np.load(f"plot_data/black_dist_box_seed_{seed}.npy")
            bx_m_blk = np.load(f"plot_data/black_mech_box_seed_{seed}.npy")
            bx_e_blk = np.load(f"plot_data/black_elec_box_seed_{seed}.npy")
            bx_k_blk = np.load(f"plot_data/black_kin_box_seed_{seed}.npy")

            bx_d_grn = np.load(f"plot_data/green_dist_box_seed_{seed}.npy")
            bx_m_grn = np.load(f"plot_data/green_mech_box_seed_{seed}.npy")
            bx_e_grn = np.load(f"plot_data/green_elec_box_seed_{seed}.npy")
            bx_k_grn = np.load(f"plot_data/green_kin_box_seed_{seed}.npy")

            bx_d_blu = np.load(f"plot_data/blue_dist_box_seed_{seed}.npy")
            bx_m_blu = np.load(f"plot_data/blue_mech_box_seed_{seed}.npy")
            bx_e_blu = np.load(f"plot_data/blue_elec_box_seed_{seed}.npy")
            bx_k_blu = np.load(f"plot_data/blue_kin_box_seed_{seed}.npy")

            # --- Apply Cumsums & Append to No Box Dict ---
            nobox['black']['dist'].append(nb_d_blk)
            nobox['black']['mech'].append(np.cumsum(nb_m_blk))
            nobox['black']['elec'].append(np.cumsum(nb_e_blk))
            nobox['black']['kin'].append(np.cumsum(nb_k_blk))
            
            nobox['blue']['dist'].append(nb_d_blu)
            nobox['blue']['mech'].append(np.cumsum(nb_m_blu))
            nobox['blue']['elec'].append(np.cumsum(nb_e_blu))
            nobox['blue']['kin'].append(np.cumsum(nb_k_blu))

            # --- Apply Cumsums & Append to With Box Dict ---
            box['black']['dist'].append(bx_d_blk)
            box['black']['mech'].append(np.cumsum(bx_m_blk))
            box['black']['elec'].append(np.cumsum(bx_e_blk))
            box['black']['kin'].append(np.cumsum(bx_k_blk))
            
            box['green']['dist'].append(bx_d_grn)
            box['green']['mech'].append(np.cumsum(bx_m_grn))
            box['green']['elec'].append(np.cumsum(bx_e_grn))
            box['green']['kin'].append(np.cumsum(bx_k_grn))
            
            box['blue']['dist'].append(bx_d_blu)
            box['blue']['mech'].append(np.cumsum(bx_m_blu))
            box['blue']['elec'].append(np.cumsum(bx_e_blu))
            box['blue']['kin'].append(np.cumsum(bx_k_blu))
            
            valid_seeds += 1

        except FileNotFoundError as e:
            print(f"  [!] Missing data for seed {seed}. Skipping... ({e})")
            continue

    if valid_seeds == 0:
        print("  [Error] No valid seeds found. Aborting.")
        return

    time_steps = np.arange(len(nobox['black']['dist'][0]))

    # ==========================================
    # 3. GRAPH SETUP & STYLING
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f"Aggregated Performance: {condition_name} (Mean ± 1 SD, n={valid_seeds})", 
                 fontsize=20, fontweight='bold', y=0.92)

    axs[0, 0].set_title("Unobstructed Environment (No Box)", fontsize=16, pad=15)
    axs[0, 1].set_title("Obstructed Environment (With Box)", fontsize=16, pad=15)

    axs[0, 0].set_ylabel("Dist. to Target (m)", fontsize=13, fontweight='bold')
    axs[1, 0].set_ylabel(r"Cumul. Mech. Work ($W = \Sigma |\tau \cdot \Delta q|$)", fontsize=13, fontweight='bold')
    axs[2, 0].set_ylabel(r"Cumul. Elec. Energy ($\Sigma \tau^2$)", fontsize=13, fontweight='bold')
    axs[3, 0].set_ylabel(r"Cumul. Kin. Energy ($\Sigma \dot{q}^2$)", fontsize=13, fontweight='bold')

    colors = {'black': '#000000', 'green': '#2CA02C', 'blue': '#1F77B4'}
    labels = {
        'black': 'Baseline (No pullback / penalties off)',
        'green': 'Penalized (No pullback / penalties active)',
        'blue':  'Pullback Architecture'
    }

    # Helper function to plot mean and std dev
    def plot_aggregated_line(ax, data_matrix, agent_color, agent_name, is_dashed=False):
        matrix = np.array(data_matrix)
        mean_val = np.mean(matrix, axis=0)
        std_val = np.std(matrix, axis=0)
        
        linestyle = '--' if is_dashed else '-'
        linewidth = 4.0 if agent_name == 'black' else 2.5
        alpha_line = 0.4 if agent_name == 'black' else 1.0
        
        ax.plot(time_steps, mean_val, color=colors[agent_name], linewidth=linewidth, 
                linestyle=linestyle, alpha=alpha_line, label=labels[agent_name])
        ax.fill_between(time_steps, mean_val - std_val, mean_val + std_val, 
                        color=colors[agent_name], alpha=0.15, edgecolor='none')

    # ==========================================
    # 4. PLOT LEFT COLUMN (No Box)
    # ==========================================
    for i, metric in enumerate(metrics):
        # Order: Black (Background), Blue (Foreground)
        plot_aggregated_line(axs[i, 0], nobox['black'][metric], 'black', 'black')
        plot_aggregated_line(axs[i, 0], nobox['blue'][metric], 'blue', 'blue')
    
    axs[0, 0].legend(loc='upper right', fontsize=11, frameon=True)

    # ==========================================
    # 5. PLOT RIGHT COLUMN (With Box)
    # ==========================================
    for i, metric in enumerate(metrics):
        # Order: Black (Background), Blue (Solid), Green (Dashed Top)
        plot_aggregated_line(axs[i, 1], box['black'][metric], 'black', 'black')
        plot_aggregated_line(axs[i, 1], box['blue'][metric], 'blue', 'blue')
        plot_aggregated_line(axs[i, 1], box['green'][metric], 'green', 'green', is_dashed=True)

    axs[0, 1].legend(loc='upper right', fontsize=11, frameon=True)

    # ==========================================
    # 6. FINAL FORMATTING
    # ==========================================
    for i in range(4):
        for j in range(2):
            axs[i, j].set_xlim(0, 500)
            axs[i, j].margins(y=0.05)
            if i == 3:
                axs[i, j].set_xlabel("Time (Steps)", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = f"plot_data/final_figures/Aggregated_4x2_{condition_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Successfully generated Final 4x2 Aggregated Figure: {save_path}")

if __name__ == "__main__":
    # Insert the 5 seeds you recorded for a specific class here
    test_seeds = [ 222,228,1296,3291,5597] 
    create_aggregated_4x2_grid(test_seeds, "Heavyweight Class")