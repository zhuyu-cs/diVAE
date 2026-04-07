import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


def create_neural_colormap():
    """Create blue-white-red colormap"""
    colors = [
        (0.0, 0.2, 0.6),
        (0.3, 0.55, 0.85),
        (0.75, 0.88, 0.98),
        (1.0, 1.0, 1.0),
        (0.99, 0.85, 0.75),
        (0.95, 0.5, 0.25),
        (0.7, 0.1, 0.1),
    ]
    return LinearSegmentedColormap.from_list('neural', colors, N=256)


def plot_original_vs_generated(data_dict, mice_list, output_path='./figures/neural_activity.pdf',
                                figsize=(16, 8)):
    """
    Plot heatmap of original vs generated neural activity.

    Layout (2 rows):
    - Row 1: original neural activity
    - Row 2: generated neural activity (with x-axis ticks)
    """
    n_mice = len(mice_list)
    
    fig = plt.figure(figsize=figsize)
    
    # Increase column spacing with wspace
    gs = gridspec.GridSpec(2, n_mice + 1, 
                           width_ratios=[1]*n_mice + [0.03],
                           height_ratios=[1, 1], 
                           hspace=0.15, wspace=0.25)
    
    cmap = create_neural_colormap()
    vmin, vmax = -2.5, 2.5
    
    im1_list = []
    im2_list = []
    
    for i, mouse in enumerate(mice_list):
        # ===== Retrieve data (unsorted) =====
        # Use continuous z-scored data
        original = data_dict[mouse]['original_continuous_zscore']  # (N, 4800)
        generated_list = data_dict[mouse]['generated_continuous_zscore_list']
        generated = generated_list[0]  # Take first repeat
        
        n_neurons = data_dict[mouse]['n_neurons']
        n_trials = data_dict[mouse]['n_trials']
        n_frames = data_dict[mouse]['n_frames_per_trial']
        
        # ===== Row 1: original data heatmap =====
        ax1 = fig.add_subplot(gs[0, i])
        im1 = ax1.imshow(original, aspect='auto', cmap=cmap, 
                         vmin=vmin, vmax=vmax, interpolation='nearest')
        im1_list.append(im1)
        
        ax1.set_title(f'Mouse {i+1}', fontsize=12, fontweight='bold')
        if i == 0:
            ax1.set_ylabel('Neurons', fontsize=11)
        ax1.set_yticks([0, n_neurons-1])
        ax1.set_yticklabels(['1', str(n_neurons)], fontsize=9)
        ax1.set_xticks([])
        
        # Neuron count annotation
        ax1.text(0.02, 0.96, f'n={n_neurons}', transform=ax1.transAxes,
                 fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        # ===== Row 2: generated data heatmap =====
        ax2 = fig.add_subplot(gs[1, i])
        im2 = ax2.imshow(generated, aspect='auto', cmap=cmap,
                         vmin=vmin, vmax=vmax, interpolation='nearest')
        im2_list.append(im2)
        
        if i == 0:
            ax2.set_ylabel('Neurons', fontsize=11)
        ax2.set_yticks([0, n_neurons-1])
        ax2.set_yticklabels(['1', str(n_neurons)], fontsize=9)
        
        # X-axis: 0-24 trials
        x_ticks = np.linspace(0, n_trials * n_frames, 5)  # 0, 6, 12, 18, 24
        x_labels = ['0', '6', '12', '18', '24']
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, fontsize=9)
        ax2.set_xlabel('CTs', fontsize=11)
    
    # ===== Colorbar: one per row =====
    # Row 1 colorbar
    cbar_ax1 = fig.add_subplot(gs[0, -1])
    cbar1 = fig.colorbar(im1_list[0], cax=cbar_ax1)
    cbar1.ax.tick_params(labelsize=9)
    
    # Row 2 colorbar
    cbar_ax2 = fig.add_subplot(gs[1, -1])
    cbar2 = fig.colorbar(im2_list[0], cax=cbar_ax2)
    cbar2.ax.tick_params(labelsize=9)
    cbar2.set_label('z-score', fontsize=10)
    
    # ===== Row labels =====
    fig.text(0.01, 0.72, 'Original', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.28, 'Generated', fontsize=12, fontweight='bold', rotation=90, va='center')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'✓ Saved: {output_path}')


if __name__ == '__main__':
    # Load processed data
    with open('./processed_data/sorted_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    
    mice = list(data_dict.keys())
    print(f"Found {len(mice)} mice: {mice}")
    
    os.makedirs('./figures', exist_ok=True)
    
    # Plot first 4 mice
    plot_original_vs_generated(data_dict, mice[:4], './figures/neural_activity.pdf')