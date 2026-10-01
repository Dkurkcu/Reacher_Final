import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_pad_data(seeds, base_path="plot_data", target_length=650):
    """
    Loads data for a single set of seeds across all three conditions and pads 
    the arrays to the target_length to show the plateau behavior clearly.
    """
    metrics = ['dist', 'mech', 'elec']
    
    nobox = {'black': {m: [] for m in metrics}, 'blue': {m: [] for m in metrics}}
    light = {'black': {m: [] for m in metrics}, 'green': {m: [] for m in metrics}}
    heavy = {'black': {m: [] for m in metrics}, 'blue': {m: [] for m in metrics}}
    
    def pad(arr):
        if len(arr) >= target_length: return arr[:target_length]
        return np.pad(arr, (0, target_length - len(arr)), 'edge')

    for seed in seeds:
        try:
            # --- Load No Box (Original Filenames) ---
            nobox['black']['dist'].append(pad(np.load(f"{base_path}/black_dist_nobox_seed_{seed}.npy")))
            nobox['black']['mech'].append(pad(np.cumsum(np.load(f"{base_path}/black_mech_nobox_seed_{seed}.npy"))))
            nobox['black']['elec'].append(pad(np.cumsum(np.load(f"{base_path}/black_elec_nobox_seed_{seed}.npy"))))
            
            nobox['blue']['dist'].append(pad(np.load(f"{base_path}/blue_dist_nobox_seed_{seed}.npy")))
            nobox['blue']['mech'].append(pad(np.cumsum(np.load(f"{base_path}/blue_mech_nobox_seed_{seed}.npy"))))
            nobox['blue']['elec'].append(pad(np.cumsum(np.load(f"{base_path}/blue_elec_nobox_seed_{seed}.npy"))))

            # --- Load Light Box (New Filenames) ---
            light['black']['dist'].append(pad(np.load(f"{base_path}/black_dist_light_seed_{seed}.npy")))
            light['black']['mech'].append(pad(np.cumsum(np.load(f"{base_path}/black_mech_light_seed_{seed}.npy"))))
            light['black']['elec'].append(pad(np.cumsum(np.load(f"{base_path}/black_elec_light_seed_{seed}.npy"))))

            light['green']['dist'].append(pad(np.load(f"{base_path}/green_dist_light_seed_{seed}.npy")))
            light['green']['mech'].append(pad(np.cumsum(np.load(f"{base_path}/green_mech_light_seed_{seed}.npy"))))
            light['green']['elec'].append(pad(np.cumsum(np.load(f"{base_path}/green_elec_light_seed_{seed}.npy"))))

            # --- Load Heavy Box (New Filenames) ---
            heavy['black']['dist'].append(pad(np.load(f"{base_path}/black_dist_heavy_seed_{seed}.npy")))
            heavy['black']['mech'].append(pad(np.cumsum(np.load(f"{base_path}/black_mech_heavy_seed_{seed}.npy"))))
            heavy['black']['elec'].append(pad(np.cumsum(np.load(f"{base_path}/black_elec_heavy_seed_{seed}.npy"))))

            heavy['blue']['dist'].append(pad(np.load(f"{base_path}/blue_dist_heavy_seed_{seed}.npy")))
            heavy['blue']['mech'].append(pad(np.cumsum(np.load(f"{base_path}/blue_mech_heavy_seed_{seed}.npy"))))
            heavy['blue']['elec'].append(pad(np.cumsum(np.load(f"{base_path}/blue_elec_heavy_seed_{seed}.npy"))))

        except FileNotFoundError as e:
            print(f"  [!] Missing data for seed {seed}. ({e})")
            continue
            
    return nobox, light, heavy

def plot_aggregated_line(ax, data_matrix, color, label=None, time_steps=np.arange(650)):
    """Helper function to plot mean and standard deviation shading."""
    matrix = np.array(data_matrix)
    mean_val = np.mean(matrix, axis=0)
    std_val = np.std(matrix, axis=0)
    
    linewidth = 4.0 if color == '#000000' else 2.5
    alpha_line = 0.4 if color == '#000000' else 1.0
    
    ax.plot(time_steps, mean_val, color=color, linewidth=linewidth, alpha=alpha_line, label=label)
    ax.fill_between(time_steps, mean_val - std_val, mean_val + std_val, color=color, alpha=0.15, edgecolor='none')

def generate_nine_individual_pdfs(nobox, light_data, heavy_data, out_dir="final_paper_figures"):
    print(f"\nBuilding 9 individual PDF plots for paper...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    time_steps = np.arange(650)
    metrics = ['dist', 'mech', 'elec']
    y_labels = {'dist': 'Distance top target [m]', 'mech': 'Mechanical Energy [J]', 'elec': 'Stall Energy Loss Index [N^2m^2]'}
    cols = ['unobstructed', 'lightweight', 'heavyweight']
    
    baseline_color, pullback_color = '#000000', '#1F77B4'

    for col_idx, col_name in enumerate(cols):
        for row_idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(5.5, 4))
            
            # 1. No Obstacle
            if col_idx == 0:
                plot_aggregated_line(ax, nobox['black'][metric], baseline_color, label='Baseline (No pullback / penalties off)')
                plot_aggregated_line(ax, nobox['blue'][metric], pullback_color, label='Pullback Architecture')
                if row_idx == 0: ax.legend(loc='upper right', fontsize=10, frameon=True)
                
            # 2. Lightweight
            elif col_idx == 1:
                plot_aggregated_line(ax, light_data['black'][metric], baseline_color)
                plot_aggregated_line(ax, light_data['green'][metric], pullback_color) # Green styled as solid blue
            # 3. Heavyweight
            elif col_idx == 2:
                plot_aggregated_line(ax, heavy_data['black'][metric], baseline_color)
                plot_aggregated_line(ax, heavy_data['blue'][metric], pullback_color)

            if metric == 'dist':
                ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
                    label='Success Tolerance (0.05m)' if (col_idx == 0 and row_idx == 0) else "")

            # Formatting
            ax.set_ylabel(y_labels[metric], fontsize=12, fontweight='bold')
            ax.set_xlim(0, 650)
            ax.margins(y=0.05)
            if row_idx == 2: ax.set_xlabel("Time (Steps)", fontsize=12)

            plt.tight_layout()
            filename = f"{out_dir}/{col_name}_{metric}.pdf"
            plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Generated: {filename}")

def generate_3x3_grid(nobox, light_data, heavy_data, out_dir="final_paper_figures"):
    print(f"\nBuilding unified 3x3 Grid plot...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    time_steps = np.arange(650)
    metrics = ['dist', 'mech', 'elec']
    y_labels = {'dist': 'Distance top target [m]', 'mech': 'Mechanical Energy [J]', 'elec': 'Stall Energy Loss Index [N^2m^2]'}
    
    baseline_color, pullback_color = '#000000', '#1F77B4'
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    # Column Titles
    axs[0, 0].set_title("Unobstructed Environment (No Box)", fontsize=16, pad=15)
    axs[0, 1].set_title("Light\nObstructed Environment (With Box)", fontsize=16, pad=15)
    axs[0, 2].set_title("Heavy\n", fontsize=16, pad=15)

    for row_idx, metric in enumerate(metrics):
        # Y-axis labels only on the far left column
        axs[row_idx, 0].set_ylabel(y_labels[metric], fontsize=14, fontweight='bold')
        
        for col_idx in range(3):
            ax = axs[row_idx, col_idx]
            
            # Plot specific column data
            if col_idx == 0:
                plot_aggregated_line(ax, nobox['black'][metric], baseline_color, label='Baseline (No pullback / penalties off)')
                plot_aggregated_line(ax, nobox['blue'][metric], pullback_color, label='Pullback Architecture')
                if row_idx == 0: ax.legend(loc='upper right', fontsize=11, frameon=True)
            elif col_idx == 1:
                plot_aggregated_line(ax, light_data['black'][metric], baseline_color)
                plot_aggregated_line(ax, light_data['green'][metric], pullback_color)
            elif col_idx == 2:
                plot_aggregated_line(ax, heavy_data['black'][metric], baseline_color)
                plot_aggregated_line(ax, heavy_data['blue'][metric], pullback_color)
                
            if metric == 'dist':
                ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
                           label='Success Tolerance (0.05m)' if (col_idx == 0 and row_idx == 0) else "")

            # Standard formatting
            ax.set_xlim(0, 650)
            ax.margins(y=0.05)
            
            # X-axis labels only on the bottom row
            if row_idx == 2:
                ax.set_xlabel("Time (Steps)", fontsize=14)

    plt.tight_layout()
    filename = f"{out_dir}/Aggregated_3x3_Grid.pdf"
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {filename}")


if __name__ == "__main__":
    # Shared seeds for all environments
    evaluation_seeds = [222, 228, 1296, 3291, 5597]
    data_directory = "plot_data"
    output_directory = "final_paper_figures"
    
    os.makedirs(output_directory, exist_ok=True)
    
    # Load and pad the data ONCE for efficiency
    print("Loading and padding data...")
    nobox_data, light_data, heavy_data = load_and_pad_data(
        seeds=evaluation_seeds, 
        base_path=data_directory, 
        target_length=650
    )
    
    # Generate the 9 individual PDFs for the paper
    generate_nine_individual_pdfs(nobox_data, light_data, heavy_data, out_dir=output_directory)
    
    # Generate the single combined 3x3 grid image
    generate_3x3_grid(nobox_data, light_data, heavy_data, out_dir=output_directory)
    
    print("\nAll plotting complete!")