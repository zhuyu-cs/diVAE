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

def get_mouse_unique_neuron_counts(mouse, all_mouse_dict, thresholds):
    """Compute the percentage of unique high-contribution neurons at different thresholds for a single mouse"""
    cur_lv = all_mouse_dict[mouse]['sild']
    
    # Get total neuron count
    first_attribution = np.array(cur_lv['0.pt']['grad'])
    first_attribution = robust_scale(first_attribution).squeeze()
    
    if len(first_attribution.shape) > 1:
        total_neurons = first_attribution.shape[0]
    else:
        total_neurons = len(first_attribution)
    
    # Store neuron coverage percentage at each threshold
    percentiles = []
    
    for threshold in thresholds:
        # Collect high-contribution neuron counts across all time points
        high_contrib_counts = []
        
        # Iterate over all 120 time points
        for idx in range(120):
            trial_key = f'{idx}.pt'
            attribution = np.array(cur_lv[trial_key]['grad'])
            attribution = robust_scale(attribution).squeeze()
            
            # Handle 2D data if present
            if len(attribution.shape) > 1:
                # Check if each neuron exceeds threshold in any dimension
                neuron_max_contrib = np.max(attribution, axis=1)  # Max contribution per neuron
                high_contrib_neurons = neuron_max_contrib > threshold
                high_contrib_count = np.sum(high_contrib_neurons)
            else:
                # Process 1D data directly
                high_contrib_count = np.sum(attribution > threshold)
            
            high_contrib_counts.append(high_contrib_count)
        
        # Compute average high-contribution neuron count and convert to percentage
        avg_count = np.mean(high_contrib_counts)
        percentage = (avg_count / total_neurons) * 100
        percentiles.append(percentage)
    
    return percentiles, total_neurons

def analyze_unique_neuron_count_thresholds_for_all_mice(all_mouse_dict, mouse_list):
    """Analyze the coverage percentage of unique high-contribution neurons at different thresholds for all mice"""
    # Threshold range
    thresholds = np.linspace(0.1, 0.8, 36)
    
    # Store neuron coverage percentage at each threshold for each mouse
    all_mice_percentiles = []
    total_neurons_list = []
    
    # Compute neuron coverage percentage for each mouse
    for mouse in mouse_list:
        print(f"Processing {mouse}...")
        mouse_percentiles, total_neurons = get_mouse_unique_neuron_counts(mouse, all_mouse_dict, thresholds)
        all_mice_percentiles.append(mouse_percentiles)
        total_neurons_list.append(total_neurons)
        print(f"  Total neurons: {total_neurons}")
    
    # Convert to numpy array
    all_mice_percentiles = np.array(all_mice_percentiles)
    
    # Compute mean and SEM
    mean_percentiles = np.mean(all_mice_percentiles, axis=0)
    sem_percentiles = stats.sem(all_mice_percentiles, axis=0)
    
    print(f"\nAverage total neurons across mice: {np.mean(total_neurons_list):.0f}")
    
    return thresholds, mean_percentiles, sem_percentiles, all_mice_percentiles

def create_style_matched_threshold_figure(thresholds, mean_percentiles, sem_percentiles):
    """Create a threshold figure with consistent style"""
    
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 13,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Use orange color
    main_color = '#ff7f0e'
    
    # Plot main curve
    ax.plot(thresholds, mean_percentiles, '-', color=main_color, linewidth=2.5)
    
    # Plot key data points
    key_indices = [0, 5, 10, 15, 20, 25, 30, 35]
    for i in key_indices:
        if i < len(thresholds):
            ax.plot(thresholds[i], mean_percentiles[i], 'o', 
                   color=main_color, markersize=6, markeredgecolor='none')
    
    # Plot SEM shaded region
    ax.fill_between(thresholds, 
                    mean_percentiles - sem_percentiles,
                    mean_percentiles + sem_percentiles,
                    color=main_color, alpha=0.2)
    
    # Find the threshold corresponding to top 5%
    target_percentage = 5.0
    closest_idx = np.argmin(np.abs(mean_percentiles - target_percentage))
    optimal_threshold = thresholds[closest_idx]
    
    # Draw horizontal reference line at 5%
    ax.axhline(y=target_percentage, color='gray', linestyle='--', alpha=0.7)
    
    # Draw vertical reference line at the corresponding threshold
    ax.axvline(x=optimal_threshold, color='gray', linestyle=':', alpha=0.7)
    
    # Set axis labels
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Percentile (Top %)')
    
    # Set axis limits
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(0, 55)
    
    # Set x-axis ticks
    ax.set_xticks([0.0, 0.4, 0.6])
    ax.set_xticklabels(['0.0', '0.4', '0.6'])
    
    # Set y-axis ticks
    ax.set_yticks([0, 25, 50])
    ax.set_yticklabels(['0', '25', '50'])
    
    # Build legend labels
    legend_text = [
        f'Top {target_percentage}%',
        f'Threshold for Top {target_percentage}%: {optimal_threshold:.3f}'
    ]
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', alpha=0.7, label=legend_text[0]),
        Line2D([0], [0], color='gray', linestyle='--', alpha=0.7, label=legend_text[1])
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=False, shadow=False, framealpha=0.9)
    
    # Apply axis formatting
    format_axis(ax)
    
    ax.grid(False)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('./fig_5b.png', 
                dpi=600, bbox_inches='tight')
    
    print(f"Top 5% neurons corresponds to threshold: {optimal_threshold:.4f}")
    print(f"Actual percentage at this threshold: {mean_percentiles[closest_idx]:.2f}%")
    
    return fig, optimal_threshold

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    with open('./all_lv_grad.pkl', "rb") as tf:
        all_mouse_dict = pickle.load(tf)
    
    mouse_list = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']
    
    # Analyze unique neuron coverage percentage at different thresholds for all mice
    thresholds, mean_percentiles, sem_percentiles, all_mice_percentiles = \
        analyze_unique_neuron_count_thresholds_for_all_mice(all_mouse_dict, mouse_list)
    
    # Create style-matched threshold figure
    fig, optimal_threshold = create_style_matched_threshold_figure(
        thresholds, mean_percentiles, sem_percentiles)