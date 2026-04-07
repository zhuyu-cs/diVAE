import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


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


def compute_global_vlim(data_list, percentile=99, clip_range=(-3, 3)):
    """Compute globally unified vmin/vmax"""
    all_values = np.concatenate([d.flatten() for d in data_list])
    abs_max = np.percentile(np.abs(all_values), percentile)
    vmin, vmax = -abs_max, abs_max
    vmin = max(vmin, clip_range[0])
    vmax = min(vmax, clip_range[1])
    return vmin, vmax


def plot_heatmap(ax, data, cmap, vmin, vmax, 
                 title='', show_xlabel=True, show_ylabel=True,
                 n_neurons_label=None, n_trials=24, frames_per_trial=200):
    """Plot a single heatmap"""
    
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper')
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    
    n_neurons, n_frames = data.shape
    
    # Y-axis
    ax.set_yticks([0, n_neurons - 1])
    ax.set_yticklabels(['1', str(n_neurons)], fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Neurons (sorted)', fontsize=10)
    
    # X-axis
    if show_xlabel:
        x_ticks = np.arange(0, n_trials + 1, 6) * frames_per_trial
        x_ticks = x_ticks[x_ticks <= n_frames]
        x_labels = [str(int(t / frames_per_trial)) for t in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Trials', fontsize=10)
    else:
        ax.set_xticks([])
    
    # Neuron count label
    if n_neurons_label is not None:
        ax.text(0.02, 0.97, f'n={n_neurons_label}', transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                          alpha=0.9, edgecolor='none'))
    
    return im


def plot_comparison_figure(data_dict, mice_list, 
                           output_path='./figures/activity_comparison.pdf',
                           figsize=(16, 7),
                           use_smooth=True):  # Whether to use smoothed data
    """
    Plot comparison figure: original on top row, generated on bottom row.
    
    Args:
        use_smooth: True uses smoothed data (better appearance), False uses raw data (more realistic)
    """
    n_mice = len(mice_list)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, n_mice + 1, 
                           width_ratios=[1]*n_mice + [0.02],
                           height_ratios=[1, 1], 
                           hspace=0.08, wspace=0.08)
    
    cmap = create_neural_colormap()
    
    # Select data fields
    if use_smooth:
        orig_key = 'original_sorted_smooth'
        gen_key = 'generated_sorted_smooth_list'
        print("Using smoothed data for visualization")
    else:
        orig_key = 'original_sorted'
        gen_key = 'generated_sorted_list'
        print("Using raw sorted data for visualization")
    
    # Collect all data to compute global vmin/vmax
    all_data = []
    for mouse in mice_list:
        all_data.append(data_dict[mouse][orig_key])
        all_data.append(data_dict[mouse][gen_key][0])
    
    vmin, vmax = compute_global_vlim(all_data, percentile=99, clip_range=(-3, 3))
    print(f"Global vmin={vmin:.2f}, vmax={vmax:.2f}")
    
    im = None
    
    # ===== Top row: original data =====
    for i, mouse in enumerate(mice_list):
        ax = fig.add_subplot(gs[0, i])
        data = data_dict[mouse][orig_key]
        n_neurons = data_dict[mouse]['n_neurons']
        n_trials = data_dict[mouse]['n_trials']
        frames_per_trial = data_dict[mouse]['n_frames_per_trial']
        
        im = plot_heatmap(ax, data, cmap, vmin, vmax,
                          title=f'Mouse {i+1}',
                          show_xlabel=False, show_ylabel=(i==0),
                          n_neurons_label=n_neurons,
                          n_trials=n_trials, frames_per_trial=frames_per_trial)
    
    # ===== Bottom row: generated data (using the same sorting order!) =====
    for i, mouse in enumerate(mice_list):
        ax = fig.add_subplot(gs[1, i])
        data = data_dict[mouse][gen_key][0]  # First repeat
        n_neurons = data_dict[mouse]['n_neurons']
        n_trials = data_dict[mouse]['n_trials']
        frames_per_trial = data_dict[mouse]['n_frames_per_trial']
        
        im = plot_heatmap(ax, data, cmap, vmin, vmax,
                          title='',
                          show_xlabel=True, show_ylabel=(i==0),
                          n_neurons_label=None,
                          n_trials=n_trials, frames_per_trial=frames_per_trial)
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('z-score', fontsize=11)
    
    # Row labels
    fig.text(0.008, 0.73, 'Original', fontsize=13, fontweight='bold', 
             rotation=90, va='center', ha='left')
    fig.text(0.008, 0.27, 'Generated', fontsize=13, fontweight='bold', 
             rotation=90, va='center', ha='left')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'✓ Saved: {output_path}')


if __name__ == '__main__':
    with open('./processed_data/sorted_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    
    mice = list(data_dict.keys())[:4]
    os.makedirs('./figures', exist_ok=True)
    
    # Plot using smoothed data (recommended, more visually consistent)
    plot_comparison_figure(data_dict, mice, 
                           './figures/activity_comparison_smooth.pdf',
                           use_smooth=True)
    
    # Plot using raw data (more realistic)
    plot_comparison_figure(data_dict, mice, 
                           './figures/activity_comparison_raw.pdf',
                           use_smooth=False)