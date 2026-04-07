import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import os

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

# Define threshold for high-contribution neurons
threshold = 0.4

def create_frequency_distribution_figure(all_mouse_dict, mouse_list):
    """Create a curve plot of neuron repetition count vs coverage percentage, based on full trial statistics"""
    # Set plot parameters
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 13,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    # Create figure
    fig = plt.figure(figsize=(6, 4))
    
    # Use autumn colormap
    autumn_colors = plt.cm.autumn([0.2, 0.5, 0.8])
    main_color = autumn_colors[1]  # Use the middle orange color
    
    # Store data for all mice
    all_mice_percentages = []
    
    # Determine maximum repetition count
    total_trials = 24  # Total of 24 complete trials
    max_frequency = total_trials  
    
    # Compute frequency distribution for each mouse
    for mouse in mouse_list:
        # Get total number of neurons
        total_neurons = len(np.array(all_mouse_dict[mouse]['sild']['0.pt']['grad']).squeeze())
        
        # Create an array to track whether each neuron is a high-contribution neuron in each trial
        neuron_in_trial = np.zeros((total_trials, total_neurons), dtype=bool)
        
        # Iterate over all 24 trials, each containing 5 consecutive time points
        for trial_idx in range(total_trials):
            # Starting time point index for the current trial
            start_idx = trial_idx * 5
            
            # If a neuron is high-contribution at any of the 5 time points, mark it as active in this trial
            for offset in range(5):
                time_idx = start_idx + offset
                trial_key = f'{time_idx}.pt'
                attribution = np.array(all_mouse_dict[mouse]['sild'][trial_key]['grad'])
                attribution = robust_scale(attribution).squeeze()
                
                # Identify high-contribution neurons
                important_indices = np.unique(np.where(attribution > threshold)[0])
                
                # Mark high-contribution neurons for the current trial
                neuron_in_trial[trial_idx, important_indices] = True
        
        # Count how many trials each neuron is a high-contribution neuron in
        neuron_frequency = np.sum(neuron_in_trial, axis=0)
        
        # Compute the percentage of neurons at each frequency (starting from 2)
        mouse_percentages = []
        for freq in range(2, max_frequency + 1):
            # Count neurons appearing at least freq times
            count = np.sum(neuron_frequency >= freq)
            percentage = (count / total_neurons) * 100
            mouse_percentages.append(percentage)
        
        all_mice_percentages.append(mouse_percentages)
    
    # Convert to numpy array
    all_mice_percentages = np.array(all_mice_percentages)
    
    # Compute mean and SEM
    mean_percentages = np.mean(all_mice_percentages, axis=0)
    sem_percentages = stats.sem(all_mice_percentages, axis=0)
    
    # Plot curve
    freq_range = np.arange(2, max_frequency + 1)
    plt.plot(freq_range, mean_percentages, color=main_color, lw=2.5)
    plt.fill_between(freq_range, 
                     mean_percentages - sem_percentages,
                     mean_percentages + sem_percentages,
                     color=main_color, alpha=0.2)
    
    # Mark specific frequency key points
    key_frequencies = [2, 5, 10, 24]
    for freq in key_frequencies:
        # Get the corresponding index (freq_range starts at 2, so index is freq-2)
        idx = freq - 2
        if idx < len(mean_percentages):  # Prevent index out of bounds
            plt.plot([freq], [mean_percentages[idx]], 'o', 
                    markersize=8, color=main_color, markeredgecolor='black')
    
    # Set axis labels
    plt.xlabel('Repeated Times', labelpad=10)
    plt.ylabel('Coverage (%)', labelpad=10)
    
    # Set x-axis range and ticks
    plt.xlim(1, max_frequency + 1)  # Start from 1 to fully display position 2
    plt.xticks([2, 5, 10, 24])
    
    # Add dashed vertical line at the last key frequency
    for freq in [key_frequencies[-1]]:
        plt.axvline(x=freq, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Format axis
    ax = plt.gca()
    format_axis(ax)
    
    # Add twin y-axis to maintain consistent style
    ax_empty = ax.twinx()
    ax_empty.set_ylim(ax.get_ylim())
    ax_empty.set_yticks([])
    format_axis(ax_empty)
    ax_empty.set_zorder(ax.get_zorder() + 1)
    
    # Annotate the value at frequency 24
    freq_24_idx = 24 - 2  # Index is 22
    freq_24_value = mean_percentages[freq_24_idx]
    freq_24_sem = sem_percentages[freq_24_idx]
    plt.axhline(y=freq_24_value, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(max_frequency-2, freq_24_value+6, 
             f"0.%", color=main_color,
             fontsize=16, va='top')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('./fig_5e.png', dpi=300, bbox_inches='tight')
    
    # Print statistics
    print("Neuron coverage percentage at key frequency points (trial-based statistics):")
    for freq in key_frequencies:
        idx = freq - 2  # Corrected index
        if idx < len(mean_percentages):
            print(f">= {freq} trials: {mean_percentages[idx]:.2f}% ± {sem_percentages[idx]:.2f}%")
    
    return fig, mean_percentages, sem_percentages

if __name__ == "__main__":
    # Load data
    with open('./all_lv_grad.pkl', "rb") as tf:
        all_mouse_dict = pickle.load(tf)

    mouse_list = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']
    
    # Create frequency distribution figure
    fig, mean_percentages, sem_percentages = create_frequency_distribution_figure(
        all_mouse_dict, mouse_list)
    
    plt.close('all')