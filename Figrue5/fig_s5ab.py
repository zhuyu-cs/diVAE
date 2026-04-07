import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle

def robust_scale(data):
    """Normalize data"""
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val+1e-8)
    return normalized

def format_axis(ax):
    """Unified axis formatting"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, direction='out')

def create_cumulative_coverage_figure(all_mouse_dict, mouse_list, key='sild', threshold=0.4):
    """Plot cumulative high-contribution neuron coverage figure (Panel 2)"""
    plt.rcParams.update({
        'font.size': 17,
        'axes.labelsize': 20,
        'axes.titlesize': 18,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 17
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create finer-grained time points
    ct_base = np.linspace(18, 41, 24)  # Base CT time points
    ct_times = np.array([])
    for ct in ct_base:
        # Add 5 subdivided time points for each CT
        ct_times = np.append(ct_times, np.linspace(ct, ct + 1, 6)[:-1])
    
    # Use autumn color scheme
    autumn_colors = plt.cm.autumn([0.2, 0.5, 0.8])
    
    # Compute cumulative coverage for each mouse
    cumulative_percentages = []
    for mouse in mouse_list:
        unique_neurons = set()
        total_neurons = len(np.array(all_mouse_dict[mouse][key]['0.pt']['grad']).squeeze())
        percentages = []
        
        for idx in range(120):  # 120 time points
            trial_key = f'{idx}.pt'
            attribution = np.array(all_mouse_dict[mouse][key][trial_key]['grad']).squeeze()
            attribution = robust_scale(attribution)
            
            hc_neurons = np.where(attribution > threshold)[0]
            unique_neurons.update(hc_neurons)
            percentages.append(len(unique_neurons) / total_neurons * 100)
        
        cumulative_percentages.append(percentages)
    
    # Compute mean and standard error
    mean_cumulative = np.mean(cumulative_percentages, axis=0)
    sem_cumulative = stats.sem(cumulative_percentages, axis=0)
    
    # Plot cumulative coverage curve
    ax.plot(ct_times, mean_cumulative, color=autumn_colors[1], lw=2.5)
    ax.fill_between(ct_times,
                    mean_cumulative - sem_cumulative,
                    mean_cumulative + sem_cumulative,
                    color=autumn_colors[1], alpha=0.2)
    
    # Get final value and add annotation
    final_value = mean_cumulative[-1]
    final_sem = sem_cumulative[-1]
    print(key, final_sem)
    idx_41 = np.argmax(ct_times)  
    # Add vertical and horizontal reference lines at CT41
    ax.axvline(x=ct_times[idx_41], color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=final_value, color='gray', linestyle='--', alpha=0.7)
    
    
    # Add final value annotation
    ax.text(ct_times[idx_41]-0.8, final_value + 12, f"{final_value:.1f}%", 
            fontsize=16, color=autumn_colors[1],
            ha='center', va='top')
    
    # Add marker at the final point
    ax.plot(ct_times[idx_41], final_value, 'o', color=autumn_colors[1], 
            markeredgecolor='black', markeredgewidth=1.5, 
            markersize=8)
    
    # Add 100% reference line
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 100)
    
    # Set axis labels and ticks
    ax.set_xlabel('Circadian Time (CT)', labelpad=10)
    ax.set_ylabel('Coverage (%)', labelpad=10)
    tick_cts = [18, 24, 30, 36, 41]
    tick_positions = []
    
    for ct in tick_cts[1:]:
        if ct == 41:
            # For CT41, use the maximum value in the data
            tick_positions.append(ct_times[idx_41])
        else:
            # For other CT points, find the rightmost point of the corresponding segment
            # Compute the corresponding index: each CT has 5 points, take the last one
            ct_idx = int((ct - 18) * 5) + 4  # Rightmost point index
            tick_positions.append(ct_times[ct_idx])
    
    # Set custom tick positions and labels
    tick_positions=[18]+tick_positions
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_cts)
    
    format_axis(ax)
    
    # Add an empty right y-axis to maintain style consistency
    ax_empty = ax.twinx()
    ax_empty.set_ylim(0, 100)
    ax_empty.set_yticks([])
    format_axis(ax_empty)
    ax_empty.set_zorder(ax.get_zorder() + 1)
    
    plt.tight_layout()
    subname='prior'
    if key=='sild':
        subname = 'both_prior'
    elif key=='ild':
        subname = 'spatial_prior'
    elif key=='pld':
        subname = 'temporal_prior'
    plt.savefig(f'./Cumulative_Coverage_{subname}.png', dpi=600, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Load data
    with open('./all_lv_grad.pkl', "rb") as tf:
        all_mouse_dict = pickle.load(tf)
    
    # Define mouse list
    mouse_list = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']
    
    # Create cumulative coverage figures
    fig = create_cumulative_coverage_figure(all_mouse_dict, mouse_list, key='sild', threshold=0.4)
    plt.close()
    fig = create_cumulative_coverage_figure(all_mouse_dict, mouse_list, key='ild', threshold=0.4)
    plt.close()
    fig = create_cumulative_coverage_figure(all_mouse_dict, mouse_list, key='pld', threshold=0.4)
    plt.close()