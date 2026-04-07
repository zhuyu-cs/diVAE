"""
Prepare neural activity data: z-score normalization, save both sorted and unsorted versions
"""

import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_original_data(mouse_name, data_path='/data/SCN/data/all_scn.pkl'):
    """Load raw neural activity data"""
    with open(data_path, 'rb') as f:
        all_scn_dict = pickle.load(f)
    
    data_3d = all_scn_dict[mouse_name]['data']  # (N, 24, 200)
    position = all_scn_dict[mouse_name]['position']
    
    n_neurons, n_trials, n_frames = data_3d.shape
    data_continuous = data_3d.reshape(n_neurons, n_trials * n_frames)
    
    return data_3d, data_continuous, position


def load_generated_data(mouse_name, generated_dir, num_repeats=5):
    """Load generated neural activity data"""
    generated_continuous_list = []
    generated_3d_list = []
    
    for repeat_idx in range(1, num_repeats + 1):
        pkl_path = os.path.join(generated_dir, f'generated_activity_repeat{repeat_idx}.pkl')
        if not os.path.exists(pkl_path):
            continue
            
        with open(pkl_path, 'rb') as f:
            generated_dict = pickle.load(f)
        
        if mouse_name in generated_dict:
            gen_continuous = generated_dict[mouse_name]['activity']
            gen_3d = gen_continuous.reshape(-1, 24, 200)
            generated_continuous_list.append(gen_continuous)
            generated_3d_list.append(gen_3d)
    
    return generated_continuous_list, generated_3d_list


def zscore_normalize(activity):
    """Z-score normalization along the time dimension"""
    original_shape = activity.shape
    activity_2d = activity.reshape(activity.shape[0], -1)
    
    mean = np.mean(activity_2d, axis=1, keepdims=True)
    std = np.std(activity_2d, axis=1, keepdims=True)
    std[std < 1e-8] = 1
    
    normalized = (activity_2d - mean) / std
    return normalized.reshape(original_shape)


def get_sort_indices(activity_continuous, smooth_sigma=50):
    """Get sort indices by neuron peak activity time"""
    smoothed = gaussian_filter1d(activity_continuous, sigma=smooth_sigma, axis=1)
    peak_times = np.argmax(smoothed, axis=1)
    sort_indices = np.argsort(peak_times)
    return sort_indices, peak_times


def prepare_all_data(mice_list, data_path, generated_dir, output_dir, num_repeats=5):
    """
    Prepare all data and save both sorted and unsorted versions.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data = {}
    
    for mouse in mice_list:
        print(f'\nProcessing {mouse}...')
        
        # Load data
        original_3d, original_continuous, position = load_original_data(mouse, data_path)
        n_neurons, n_trials, n_frames = original_3d.shape
        
        generated_mouse = mouse.replace('all_data_', 'all_neuron_') if mouse.startswith('all_data_') else mouse
        generated_continuous_list, generated_3d_list = load_generated_data(
            generated_mouse, generated_dir, num_repeats
        )
        
        if len(generated_continuous_list) == 0:
            print(f"  Warning: No generated data found, skipping...")
            continue
        
        # Check neuron count
        gen_n_neurons = generated_continuous_list[0].shape[0]
        if gen_n_neurons != n_neurons:
            min_n = min(n_neurons, gen_n_neurons)
            original_3d = original_3d[:min_n]
            original_continuous = original_continuous[:min_n]
            position = position[:min_n] if position is not None else None
            generated_continuous_list = [g[:min_n] for g in generated_continuous_list]
            generated_3d_list = [g[:min_n] for g in generated_3d_list]
            n_neurons = min_n
        
        # Z-score normalization
        original_continuous_zscore = zscore_normalize(original_continuous)
        original_3d_zscore = original_continuous_zscore.reshape(n_neurons, n_trials, n_frames)
        
        generated_continuous_zscore_list = [zscore_normalize(g) for g in generated_continuous_list]
        generated_3d_zscore_list = [g.reshape(n_neurons, n_trials, n_frames) 
                                     for g in generated_continuous_zscore_list]
        
        # Get sort indices
        sort_indices, peak_times = get_sort_indices(original_continuous_zscore)
        
        # Sorted data
        original_sorted = original_continuous_zscore[sort_indices]
        generated_sorted_list = [g[sort_indices] for g in generated_continuous_zscore_list]
        
        all_data[mouse] = {
            # Unsorted (used for correlation computation)
            'original_3d_zscore': original_3d_zscore,
            'original_continuous_zscore': original_continuous_zscore,
            'generated_3d_zscore_list': generated_3d_zscore_list,
            'generated_continuous_zscore_list': generated_continuous_zscore_list,
            
            # Sorted (used for visualization)
            'original_sorted': original_sorted,
            'generated_sorted_list': generated_sorted_list,
            
            # Metadata
            'sort_indices': sort_indices,
            'peak_times': peak_times,
            'position': position,
            'n_neurons': n_neurons,
            'n_trials': n_trials,
            'n_frames_per_trial': n_frames,
        }
        
        print(f"  ✓ Neurons: {n_neurons}, Repeats: {len(generated_3d_list)}")
    
    # Save
    output_path = os.path.join(output_dir, 'sorted_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f'\n✓ Saved to: {output_path}')
    return all_data


if __name__ == '__main__':
    all_mice = [ 'SCN1', 'SCN2', 'SCN3', 'SCN4' ]
    
    prepare_all_data(
        mice_list=all_mice,
        data_path='../../data/all_scn.pkl',
        generated_dir='../GenerateSCN_data/generated_activity',
        output_dir='./processed_data',
        num_repeats=5,
    )