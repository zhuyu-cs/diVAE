import matplotlib.contour
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pickle
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d

mouse_list = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']

def nature_style():
    nature_style = {
        "font.family": "Arial",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titlesize": 12,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "lines.linewidth": 2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.figsize": [8, 8],
        "figure.dpi": 100,
        "figure.autolayout": True,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c5128_f32b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]),
    }
    plt.style.use(nature_style)

def robust_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val+1e-8)
    return normalized

# Load data
with open(f'./all_lv_grad.pkl', "rb") as tf:
    all_mouse_dict = pickle.load(tf)

top_percent = 0.05  # Select top 5% of neurons

for mouse in mouse_list:
    mouse_id = 1 if '1' in mouse else None
    train_sild_ild = all_mouse_dict[mouse]
    
    for key in ['sild']:
        name = 'biLV' if key == 'sild' else None
        cur_lv = train_sild_ild[key]
        
        # Process each CT time point
        for ct_index in range(24):  # 24 CT time points
            ct_number = ct_index + 18  # CT starts from 18
            print(f'Processing CT{ct_number}')
            
            # Create figure
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.patch.set_alpha(0)

            # Set viewing angle and background
            if mouse == 'SCN1':
                ax.view_init(elev=-90, azim=-90, roll=-5)
            else:
                ax.view_init(elev=-90, azim=-90)
            ax.set_facecolor('white')
            
            # Define circular boundary parameters
            circle_center_x = 320 
            circle_center_y = 350
            circle_radius = 340
            
            # Draw filled circular background
            circle_filled = plt.Circle((circle_center_x, circle_center_y), circle_radius, 
                                       color='lightsteelblue', 
                                       alpha=0.2,
                                       transform=ax.transData._b)
            ax.add_patch(circle_filled)
            art3d.pathpatch_2d_to_3d(circle_filled, z=0, zdir="z")
            
            # Get neuron coordinates from the first subtrial
            first_idx = ct_index * 5
            first_trial = f"{first_idx}.pt"
            coords = np.array(cur_lv[first_trial]['pos']).squeeze()
            total_neurons = coords.shape[0]
            
            # Collect weights from all subtrials under the current CT
            all_weights = []
            
            # Process the 5 subtrials within the current CT
            for sub_idx in range(5):
                idx = ct_index * 5 + sub_idx
                trial = f"{idx}.pt"
                
                # Process attribution data
                attribution_map = np.array(cur_lv[trial]['grad'])
                attribution_map = robust_scale(attribution_map)
                weights = attribution_map.squeeze()
                
                # Handle 2D weight arrays by taking max across time axis
                if len(weights.shape) > 1:
                    weights = np.max(weights, axis=1)
                
                all_weights.append(weights)
            
            # Aggregate all weights and take the maximum per neuron
            combined_weights = np.array(all_weights)  # shape: (5, n_neurons)
            max_weights = np.max(combined_weights, axis=0)  # Max weight per neuron
            
            # Compute the number of top 5% neurons
            top_count = max(1, int(total_neurons * top_percent))
            
            # Get indices of top 5% neurons
            top_indices = np.argsort(max_weights)[-top_count:]
            
            print(f"  Total neurons: {total_neurons}, Top {top_percent*100}%: {top_count}")
            
            # Plot top neurons if any exist
            if len(top_indices) > 0:
                top_coords = coords[top_indices]
                
                # Normalize CT index to color value in range [0, 1]
                color_value = ct_index / 23.0
                
                # Plot top neurons using autumn colormap mapped to current CT time point
                ax.scatter(top_coords[:, 0], 
                           top_coords[:, 1], 
                           top_coords[:, 2], 
                           c=np.full(len(top_coords), color_value),
                           cmap='autumn',
                           vmin=0,
                           vmax=1,
                           edgecolors='none', 
                           s=40,
                           alpha=0.8)
            
            # Plot remaining non-top neurons in gray
            other_mask = np.ones(coords.shape[0], dtype=bool)
            other_mask[top_indices] = False
            other_indices = np.where(other_mask)[0]
            
            if len(other_indices) > 0:
                other_coords = coords[other_indices]
                ax.scatter(other_coords[:, 0], 
                           other_coords[:, 1], 
                           other_coords[:, 2], 
                           color='gray', 
                           edgecolors='none', 
                           alpha=0.1,
                           s=20)
            
            # Set axis limits with margin
            data_min = np.min(coords, axis=0)
            data_max = np.max(coords, axis=0)
            margin = 100  # Add margin around data bounds
            
            ax.set_xlim(data_min[0] - margin, data_max[0] + margin)
            ax.set_ylim(data_min[1] - margin, data_max[1] + margin)
            ax.set_zlim(data_min[2] - margin, data_max[2] + margin)
            
            # Remove axes and grid
            ax.axis('off')
            ax.grid(False)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Create output directory if not exists
            os.makedirs(f'./Figure_5c/{key}_top5percent', exist_ok=True)
            
            # Save figure
            save_filename = f"{mouse.split('-')[0]}_CT{ct_number}_top5percent"
            plt.savefig(
                f'./Figure_5c/{key}_top5percent/{save_filename}.png',
                dpi=600,
                bbox_inches='tight',
                transparent=True,
                facecolor='none',
                edgecolor='none'
            )
            
            plt.close()

print("Processing complete!")