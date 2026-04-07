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
    """Uniform axis formatting"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, direction='out')

def create_ct_overlap_figure(all_mouse_dict, mouse_list, optimal_thresholds, mode_pair=('pld', 'ild')):
    """Plot overlap ratio between two modes across CT time points, displayed as percentage"""
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
    
    # Get modes and thresholds
    mode1, mode2 = mode_pair
    threshold1 = optimal_thresholds[mode1]
    threshold2 = optimal_thresholds[mode2]
    
    # Create CT time points
    ct_base = np.linspace(18, 41, 24)  # Base CT time points (24 hours)
    
    # Store overlap ratio for each CT time point
    overlap_by_ct = {ct: [] for ct in ct_base}
    
    # Compute overlap ratio for each mouse at each CT time point
    for mouse in mouse_list:
        for ct_idx, ct in enumerate(ct_base):
            # Get the 5 trials corresponding to this CT
            ct_overlap_ratios = []
            
            for sub_idx in range(5):
                idx = ct_idx * 5 + sub_idx
                trial_key = f'{idx}.pt'
                
                if trial_key in all_mouse_dict[mouse][mode1] and trial_key in all_mouse_dict[mouse][mode2]:
                    # Get and normalize attribution scores
                    attribution1 = np.array(all_mouse_dict[mouse][mode1][trial_key]['grad'])
                    attribution2 = np.array(all_mouse_dict[mouse][mode2][trial_key]['grad'])
                    
                    normalized_attr1 = robust_scale(attribution1)
                    normalized_attr2 = robust_scale(attribution2)
                    
                    # If temporal dimension exists, take max across time axis
                    if normalized_attr1.ndim > 1:
                        max_attr1 = np.max(normalized_attr1, axis=1)
                    else:
                        max_attr1 = normalized_attr1
                        
                    if normalized_attr2.ndim > 1:
                        max_attr2 = np.max(normalized_attr2, axis=1)
                    else:
                        max_attr2 = normalized_attr2
                    
                    # Identify important neurons above threshold
                    important_neurons1 = max_attr1 > threshold1
                    important_neurons2 = max_attr2 > threshold2
                    
                    # Compute overlap ratio (Jaccard index)
                    intersection = np.sum(important_neurons1 & important_neurons2)
                    union = np.sum(important_neurons1 | important_neurons2)
                    
                    if union > 0:  # Avoid division by zero
                        overlap_ratio = intersection / union
                        ct_overlap_ratios.append(overlap_ratio)
            
            # If data exists for this CT, store the mean overlap ratio for this mouse
            if ct_overlap_ratios:
                overlap_by_ct[ct].append(np.mean(ct_overlap_ratios))
    
    # Compute mean and SEM for each CT bin
    ct_means = []
    ct_sems = []
    
    for ct in ct_base:
        if overlap_by_ct[ct]:
            ct_means.append(np.mean(overlap_by_ct[ct]))
            ct_sems.append(stats.sem(overlap_by_ct[ct]))
        else:
            ct_means.append(np.nan)
            ct_sems.append(np.nan)
    
    # Convert overlap ratios to percentage
    ct_means_percent = [x * 100 for x in ct_means]
    ct_sems_percent = [x * 100 for x in ct_sems]
    
    # Color mapping for each mode pair
    color_map = {
        ('sild', 'pld'): plt.cm.autumn(0.5),  # Orange
        ('sild', 'ild'): plt.cm.autumn(0.7),  # Yellow-orange
        ('pld', 'ild'): plt.cm.autumn(0.3)    # Red
    }
    
    # Use default color if mode pair is not in the color map
    if mode_pair in color_map:
        color = color_map[mode_pair]
    else:
        color = plt.cm.autumn(0.6)  # Default color
    
    # Plot overlap ratio curve (in percentage)
    ax.plot(ct_base, ct_means_percent, '-', color=color, lw=2.5)
    ax.fill_between(ct_base,
                    np.array(ct_means_percent) - np.array(ct_sems_percent),
                    np.array(ct_means_percent) + np.array(ct_sems_percent),
                    color=color, alpha=0.2)
    
    # Compute overall mean overlap ratio (in percentage)
    valid_means = [m for m in ct_means_percent if not np.isnan(m)]
    mean_overlap_percent = np.mean(valid_means) if valid_means else 0
    ax.axhline(y=mean_overlap_percent, color='gray', linestyle='--', alpha=0.7)
    
    # Annotate mean overlap ratio on the plot
    ax.text(40, mean_overlap_percent + 3, f"{mean_overlap_percent:.1f}%", 
            fontsize=16, color='gray', ha='right', va='bottom')
    
    # Set axis labels and ticks
    ax.set_xlabel('Circadian Time (CT)', labelpad=10)
    ax.set_ylabel('Overlap Ratio (%)', labelpad=10)
    
    # Set x-axis ticks
    tick_cts = [18, 24, 30, 36, 41]
    ax.set_xticks(tick_cts)
    ax.set_xticklabels(tick_cts)
    
    # Set y-axis range to 0-100% to emphasize low overlap ratio
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Add 100% reference line
    ax.axhline(y=100, color='lightgray', linestyle=':', alpha=0.5)
    
    # Mode name mapping
    mode_names = {
        'pld': 'Circadian-prior',
        'ild': 'Position-prior'
    }
    
    # Apply axis formatting
    format_axis(ax)
    
    # Add empty twin y-axis to maintain consistent style
    ax_empty = ax.twinx()
    ax_empty.set_ylim(0, 100)
    ax_empty.set_yticks([])
    format_axis(ax_empty)
    ax_empty.set_zorder(ax.get_zorder() + 1)
    
    plt.tight_layout()
    output_path = f'./Spatial_Overlap_spatial_temporal_prior.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    
    return fig, overlap_by_ct

if __name__ == "__main__":
    # Load data
    with open('./all_lv_grad.pkl', "rb") as tf:
        all_mouse_dict = pickle.load(tf)
    
    # Define mouse list
    mouse_list = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']
    # Set optimal thresholds
    optimal_thresholds = {
        'pld': 0.4,
        'ild': 0.4
    }
    
    # Define all mode pairs to analyze
    mode_pairs = [ ('pld', 'ild')]
    
    # Store results for all mode pairs
    all_results = {}
    
    # Generate overlap ratio vs CT plot for each mode pair
    for mode_pair in mode_pairs:
        print(f"\nAnalyzing overlap between {mode_pair[0].upper()} and {mode_pair[1].upper()}...")
        fig, overlap_data = create_ct_overlap_figure(all_mouse_dict, mouse_list, optimal_thresholds, mode_pair=mode_pair)
        
        # Compute and print overall mean overlap ratio
        all_overlap_values = []
        for ct, values in overlap_data.items():
            if values:
                all_overlap_values.extend(values)
        
        mean_overlap = np.mean(all_overlap_values) if all_overlap_values else 0
        sem_overlap = stats.sem(all_overlap_values) if len(all_overlap_values) > 1 else 0
        
        print(f"Overall mean overlap between {mode_pair[0].upper()} and {mode_pair[1].upper()}: "
              f"{mean_overlap*100:.1f}% ± {sem_overlap*100:.1f}%")
        
        # Check if any CT time point has overlap ratio significantly above or below the mean
        significant_cts = []
        for ct, values in overlap_data.items():
            if values and len(values) >= 3:
                ct_mean = np.mean(values)
                if abs(ct_mean - mean_overlap) > 2 * sem_overlap:  # Use 2x SEM as significance criterion
                    significant_cts.append((ct, ct_mean*100))
        
        if significant_cts:
            print("CT time points significantly different from the mean:")
            for ct, ct_mean in significant_cts:
                print(f"  CT{ct:.1f}: {ct_mean:.1f}% ({'above' if ct_mean > mean_overlap*100 else 'below'} average)")
        
        # Save results
        all_results[mode_pair] = {
            'overlap_data': overlap_data,
            'mean': mean_overlap,
            'sem': sem_overlap
        }
        
        plt.close()
    
    # Compare mean overlap ratios across mode pairs
    print("\nMode pair overlap ratio summary:")
    for mode_pair, result in all_results.items():
        print(f"{mode_pair[0].upper()}-{mode_pair[1].upper()}: {result['mean']*100:.1f}% ± {result['sem']*100:.1f}%")