import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


def compute_trial_correlation(original_3d, generated_3d):
    """Compute per-trial Pearson correlation coefficients"""
    n_trials = original_3d.shape[1]
    correlations = np.zeros(n_trials)
    
    for t in range(n_trials):
        orig = original_3d[:, t, :].flatten()
        gen = generated_3d[:, t, :].flatten()
        if np.std(orig) > 1e-8 and np.std(gen) > 1e-8:
            correlations[t], _ = pearsonr(orig, gen)
    
    return correlations


def create_neural_colormap():
    """Blue-white-red colormap"""
    colors = [
        (0.0, 0.2, 0.6), (0.3, 0.55, 0.85), (0.75, 0.88, 0.98),
        (1.0, 1.0, 1.0),
        (0.99, 0.85, 0.75), (0.95, 0.5, 0.25), (0.7, 0.1, 0.1),
    ]
    return LinearSegmentedColormap.from_list('neural', colors, N=256)


def plot_correlation_with_std(ax, trials, mean_corrs, std_corrs, color='#2E86AB'):
    """
    Plot correlation curve with multi-layer std fill.

    Effect: dark 1-std band + light 2-std band + main line
    """
    # 2-std region (lightest)
    ax.fill_between(trials, 
                    mean_corrs - 2*std_corrs, 
                    mean_corrs + 2*std_corrs,
                    alpha=0.15, color=color, linewidth=0)
    
    # 1-std region (darker)
    ax.fill_between(trials, 
                    mean_corrs - std_corrs, 
                    mean_corrs + std_corrs,
                    alpha=0.3, color=color, linewidth=0)
    
    # Main line
    ax.plot(trials, mean_corrs, '-', color=color, linewidth=2)


def plot_combined_figure(data_dict, mice_list, output_path, use_sorted=False):
    """
    Plot combined figure (3 rows x N columns)
    """
    n_mice = len(mice_list)
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        3, n_mice + 1,
        width_ratios=[1]*n_mice + [0.02],
        height_ratios=[1, 1, 0.5],
        hspace=0.25, wspace=0.35
    )
    
    cmap = create_neural_colormap()
    row_ims = {0: [], 1: []}
    row_vranges = {0: [], 1: []}
    
    # Select data keys
    if use_sorted:
        orig_key = 'original_sorted'
        gen_key = 'generated_sorted_list'
    else:
        orig_key = 'original_continuous_zscore'
        gen_key = 'generated_continuous_zscore_list'
    
    # Compute global vmin/vmax per row
    for mouse in mice_list:
        orig = data_dict[mouse][orig_key]
        gen = data_dict[mouse][gen_key][0]
        row_vranges[0].append((np.percentile(orig, 2), np.percentile(orig, 98)))
        row_vranges[1].append((np.percentile(gen, 2), np.percentile(gen, 98)))
    
    vmin_row0 = min(v[0] for v in row_vranges[0])
    vmax_row0 = max(v[1] for v in row_vranges[0])
    vmin_row1 = min(v[0] for v in row_vranges[1])
    vmax_row1 = max(v[1] for v in row_vranges[1])
    
    for i, mouse in enumerate(mice_list):
        original = data_dict[mouse][orig_key]
        generated = data_dict[mouse][gen_key][0]
        original_3d = data_dict[mouse]['original_3d_zscore']
        
        n_neurons = data_dict[mouse]['n_neurons']
        n_trials = data_dict[mouse]['n_trials']
        
        # Row 1: original data
        ax1 = fig.add_subplot(gs[0, i])
        im1 = ax1.imshow(original, aspect='auto', cmap=cmap,
                         vmin=vmin_row0, vmax=vmax_row0, interpolation='nearest')
        row_ims[0].append(im1)
        
        ax1.set_title(f'SCN {i+1}', fontsize=16, fontweight='bold')
        if i == 0:
            ylabel = 'Real Neurons' if use_sorted else 'Neurons'
            ax1.set_ylabel(ylabel, fontsize=14)
        ax1.set_yticks([0, n_neurons-1])
        ax1.set_yticklabels(['1', str(n_neurons)], fontsize=12)
        ax1.set_xticks([])
        
        # Row 2: generated data
        ax2 = fig.add_subplot(gs[1, i])
        im2 = ax2.imshow(generated, aspect='auto', cmap=cmap,
                         vmin=vmin_row1, vmax=vmax_row1, interpolation='nearest')
        row_ims[1].append(im2)
        
        if i == 0:
            ylabel = 'Generated Neurons' if use_sorted else 'Neurons'
            ax2.set_ylabel(ylabel, fontsize=14)
        ax2.set_yticks([0, n_neurons-1])
        ax2.set_yticklabels(['1', str(n_neurons)], fontsize=12)
        ax2.set_xticks([])
        
        # Row 3: correlation curves (mean ± std across 5 repeats)
        ax3 = fig.add_subplot(gs[2, i])
        
        all_corrs = []
        for gen_3d in data_dict[mouse]['generated_3d_zscore_list']:
            corrs = compute_trial_correlation(original_3d, gen_3d)
            all_corrs.append(corrs)
        
        all_corrs = np.array(all_corrs)  # (5, 24)
        mean_corrs = np.mean(all_corrs, axis=0)
        std_corrs = np.std(all_corrs, axis=0)
        
        trials = np.arange(n_trials)
        
        # Plot with multi-layer fill
        plot_correlation_with_std(ax3, trials, mean_corrs, std_corrs, color='#2E86AB')
        
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax3.set_ylim(-0.1, 0.6)
        ax3.set_xlim(0, n_trials - 1)
        ax3.set_xticks(np.arange(0, n_trials + 1, 6))
        ax3.set_xlabel('CTs', fontsize=14)
        if i == 0:
            ax3.set_ylabel('Pearson r', fontsize=14)
        
    # Colorbar
    cbar_ax1 = fig.add_subplot(gs[0, -1])
    cbar1 = fig.colorbar(row_ims[0][0], cax=cbar_ax1)
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_label('z-score', fontsize=12)
    
    cbar_ax2 = fig.add_subplot(gs[1, -1])
    cbar2 = fig.colorbar(row_ims[1][0], cax=cbar_ax2)
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_label('z-score', fontsize=12)
    
    # Row labels
    fig.text(0.01, 0.78, 'Original', fontsize=16, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.48, 'Generated', fontsize=16, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.15, 'Correlation', fontsize=16, fontweight='bold', rotation=90, va='center')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'✓ Saved: {output_path}')


if __name__ == '__main__':
    with open('./processed_data/sorted_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    
    mice = list(data_dict.keys())[:4]
    os.makedirs('./figures', exist_ok=True)
    
    plot_combined_figure(data_dict, mice, './figures/generation_quality_unsorted.pdf', use_sorted=False)
    plot_combined_figure(data_dict, mice, './figures/generation_quality_sorted.pdf', use_sorted=True)