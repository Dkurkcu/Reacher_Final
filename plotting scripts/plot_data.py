import numpy as np
import matplotlib.pyplot as plt
import os

def create_4panel_ablation_grids():
    # Update with your current seeds
    seeds = [222,228,1567,2243,3311] 
    os.makedirs("plot_data/final_figures", exist_ok=True)

    for seed in seeds:
        print(f"\nBuilding 4-Panel Grid for Seed {seed}...")
        
        try:
            # ==========================================
            # 1. LOAD LEFT COLUMN (No Box - 2 Lines)
            # ==========================================
            dist_nobox_black = np.load(f"plot_data/black_dist_nobox_seed_{seed}.npy")
            mech_nobox_black = np.load(f"plot_data/black_mech_nobox_seed_{seed}.npy")
            elec_nobox_black = np.load(f"plot_data/black_elec_nobox_seed_{seed}.npy")
            kin_nobox_black  = np.load(f"plot_data/black_kin_nobox_seed_{seed}.npy")
            
            dist_nobox_blue = np.load(f"plot_data/blue_dist_nobox_seed_{seed}.npy")
            mech_nobox_blue = np.load(f"plot_data/blue_mech_nobox_seed_{seed}.npy")
            elec_nobox_blue = np.load(f"plot_data/blue_elec_nobox_seed_{seed}.npy")
            kin_nobox_blue  = np.load(f"plot_data/blue_kin_nobox_seed_{seed}.npy")

            # ==========================================
            # 2. LOAD RIGHT COLUMN (With Box - 3 Lines)
            # ==========================================
            dist_box_black = np.load(f"plot_data/black_dist_box_seed_{seed}.npy")
            mech_box_black = np.load(f"plot_data/black_mech_box_seed_{seed}.npy")
            elec_box_black = np.load(f"plot_data/black_elec_box_seed_{seed}.npy")
            kin_box_black  = np.load(f"plot_data/black_kin_box_seed_{seed}.npy")

            dist_box_green = np.load(f"plot_data/green_dist_box_seed_{seed}.npy")
            mech_box_green = np.load(f"plot_data/green_mech_box_seed_{seed}.npy")
            elec_box_green = np.load(f"plot_data/green_elec_box_seed_{seed}.npy")
            kin_box_green  = np.load(f"plot_data/green_kin_box_seed_{seed}.npy")

            dist_box_blue = np.load(f"plot_data/blue_dist_box_seed_{seed}.npy")
            mech_box_blue = np.load(f"plot_data/blue_mech_box_seed_{seed}.npy")
            elec_box_blue = np.load(f"plot_data/blue_elec_box_seed_{seed}.npy")
            kin_box_blue  = np.load(f"plot_data/blue_kin_box_seed_{seed}.npy")
            
        except FileNotFoundError as e:
            print(f"  [!] Missing data for seed {seed}. Ensure all scenarios are recorded. ({e})")
            continue

        time_steps = np.arange(len(dist_nobox_black))

        # ==========================================
        # 3. APPLY CUMULATIVE SUM TO ENERGY METRICS
        # ==========================================
        mech_nobox_black = np.cumsum(mech_nobox_black)
        elec_nobox_black = np.cumsum(elec_nobox_black)
        kin_nobox_black  = np.cumsum(kin_nobox_black)
        
        mech_nobox_blue = np.cumsum(mech_nobox_blue)
        elec_nobox_blue = np.cumsum(elec_nobox_blue)
        kin_nobox_blue  = np.cumsum(kin_nobox_blue)

        mech_box_black = np.cumsum(mech_box_black)
        elec_box_black = np.cumsum(elec_box_black)
        kin_box_black  = np.cumsum(kin_box_black)
        
        mech_box_green = np.cumsum(mech_box_green)
        elec_box_green = np.cumsum(elec_box_green)
        kin_box_green  = np.cumsum(kin_box_green)
        
        mech_box_blue = np.cumsum(mech_box_blue)
        elec_box_blue = np.cumsum(elec_box_blue)
        kin_box_blue  = np.cumsum(kin_box_blue)

        # ==========================================
        # 4. ROBUST COLLISION DETECTION
        # ==========================================
        collision_step = None
        
        # Check where Blue explicitly pulls back
        dist_diffs = np.diff(dist_box_blue)
        pullback_indices = np.where(dist_diffs > 0.001)[0]
        
        if len(pullback_indices) > 0:
            collision_step = pullback_indices[0]
        else:
            # Fallback check: Where Black stalled
            black_diffs = np.abs(np.diff(dist_box_black))
            stall_indices = np.where(black_diffs < 0.0001)[0]
            valid_stalls = [idx for idx in stall_indices if 10 < idx < 400]
            if len(valid_stalls) > 0:
                collision_step = valid_stalls[0]

        # ==========================================
        # 5. GRAPH SETUP & STYLING
        # ==========================================
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(f"Algorithm Performance & Safety Profile (Seed {seed})", 
                     fontsize=20, fontweight='bold', y=0.92)

        axs[0, 0].set_title("Unobstructed Environment (No Box)", fontsize=16, pad=15)
        axs[0, 1].set_title("Obstructed Environment (With Box)", fontsize=16, pad=15)

        axs[0, 0].set_ylabel("Dist. to Target (m)", fontsize=13, fontweight='bold')
        axs[1, 0].set_ylabel(r"Mech. Work ($W = \Sigma |\tau \cdot \Delta q|$)", fontsize=13, fontweight='bold')
        axs[2, 0].set_ylabel(r"Elec. Energy ($\Sigma \tau^2$)", fontsize=13, fontweight='bold')
        axs[3, 0].set_ylabel(r"Kin. Energy ($\Sigma \dot{q}^2$)", fontsize=13, fontweight='bold')

        c_black = '#000000' 
        c_green = '#2CA02C' 
        c_blue  = '#1F77B4' 

        # ==========================================
        # 6. PLOT LEFT COLUMN (No Box)
        # ==========================================
        axs[0, 0].plot(time_steps, dist_nobox_black, color=c_black, linewidth=4.0, alpha=0.35, label='No pullback / penalties off')
        axs[0, 0].plot(time_steps, dist_nobox_blue, color=c_blue, linewidth=2.5, label='Pullback algorithm')
        
        axs[1, 0].plot(time_steps, mech_nobox_black, color=c_black, linewidth=4.0, alpha=0.35)
        axs[1, 0].plot(time_steps, mech_nobox_blue, color=c_blue, linewidth=2.5)
        
        axs[2, 0].plot(time_steps, elec_nobox_black, color=c_black, linewidth=4.0, alpha=0.35)
        axs[2, 0].plot(time_steps, elec_nobox_blue, color=c_blue, linewidth=2.5)
        
        axs[3, 0].plot(time_steps, kin_nobox_black, color=c_black, linewidth=4.0, alpha=0.35)
        axs[3, 0].plot(time_steps, kin_nobox_blue, color=c_blue, linewidth=2.5)

        axs[0, 0].legend(loc='upper right', fontsize=11, frameon=True)

        # ==========================================
        # 7. PLOT RIGHT COLUMN (With Box) - FIXED ECLIPSE
        # ==========================================
        # Note the order here: Black (Background) -> Blue (Solid) -> Green (Dashed on top)
        axs[0, 1].plot(time_steps, dist_box_black, color=c_black, linewidth=4.0, alpha=0.35, label='No pullback / penalties off')
        axs[0, 1].plot(time_steps, dist_box_blue, color=c_blue, linewidth=2.5, label='Pullback algorithm')
        axs[0, 1].plot(time_steps, dist_box_green, color=c_green, linewidth=2.5, linestyle='--', label='No pullback / penalties active')
        
        axs[1, 1].plot(time_steps, mech_box_black, color=c_black, linewidth=4.0, alpha=0.35)
        axs[1, 1].plot(time_steps, mech_box_blue, color=c_blue, linewidth=2.5)
        axs[1, 1].plot(time_steps, mech_box_green, color=c_green, linewidth=2.5, linestyle='--')
        
        axs[2, 1].plot(time_steps, elec_box_black, color=c_black, linewidth=4.0, alpha=0.35)
        axs[2, 1].plot(time_steps, elec_box_blue, color=c_blue, linewidth=2.5)
        axs[2, 1].plot(time_steps, elec_box_green, color=c_green, linewidth=2.5, linestyle='--')
        
        axs[3, 1].plot(time_steps, kin_box_black, color=c_black, linewidth=4.0, alpha=0.35)
        axs[3, 1].plot(time_steps, kin_box_blue, color=c_blue, linewidth=2.5)
        axs[3, 1].plot(time_steps, kin_box_green, color=c_green, linewidth=2.5, linestyle='--')

        axs[0, 1].legend(loc='upper right', fontsize=11, frameon=True)

        # ==========================================
        # 8. BULLETPROOF COLLISION LINES
        # ==========================================
        if collision_step is not None:
            for i in range(4):
                axs[i, 1].axvline(x=collision_step, color='#808080', linestyle=':', linewidth=2.0)
            
            axs[0, 1].text(collision_step + 5, 0.85, 'Collision & Pullback', 
                           transform=axs[0, 1].get_xaxis_transform(),
                           color='#606060', style='italic', fontweight='bold', fontsize=10)

        # ==========================================
        # 9. FINAL FORMATTING
        # ==========================================
        for i in range(4):
            for j in range(2):
                axs[i, j].set_xlim(0, 500)
                axs[i, j].margins(y=0.05)
                if i == 3:
                    axs[i, j].set_xlabel("Time (Steps)", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_path = f"plot_data/final_figures/Final_4Panel_Grid_Seed_{seed}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  -> Generated: {save_path} (Collision Detected at step {collision_step})")

if __name__ == "__main__":
    create_4panel_ablation_grids()