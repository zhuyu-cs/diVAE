import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pickle
from sklearn.manifold import TSNE
from matplotlib.ticker import MultipleLocator, MaxNLocator
import random


def set_all_seeds(seed):
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Set CUDA environment
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)

# ── Config ──────────────────────────────────────────────────────────────────
config = {
    'latent_methods': {
        'VAE': {
            'pkl_path': '../fig2a_classifications/latents/vae.pkl',
            'feature_type': 'pld'
        },
        'piVAE-S': {
            'pkl_path': '../fig2a_classifications/latents/pivae_s.pkl',
            'feature_type': 'pld'
        },
        'piVAE-T': {
            'pkl_path': '../fig2a_classifications/latents/pivae_t.pkl',
            'feature_type': 'pld'
        },
        'diVAE': {
            'pkl_path': '../fig2a_classifications/latents/divae.pkl',
            'feature_type': 'latent_variable'
        }
    },
    'include_baselines': True
}

mice_train = ['SCN1','SCN2','SCN3','SCN4']

mice_val=['SCN5','SCN6']

output_dir = './resutls/'
os.makedirs(output_dir, exist_ok=True)


# ── Axis helper ──────────────────────────────────────────────────────────────
def tidy_axis(ax, top=False, right=False, left=False, bottom=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    ax.xaxis.set_tick_params(top='off', direction='out', width=0.5)
    ax.yaxis.set_tick_params(right='off', left='off', direction='out', width=0.5)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# ── Colormap ─────────────────────────────────────────────────────────────────
def create_gradient_colormap(n_trials=24, n_sub_points=5):
    # Main colors from RdYlBu_r palette
    colors_list = [
        '#2166AC',  # dark blue
        '#92C5DE',  # light blue
        '#F4D03F',  # yellow
        '#F2635C',  # light red
        '#B91D1D'   # dark red
    ]

    # Positions for smooth color transitions
    positions = [0, 0.25, 0.5, 0.75, 1]

    # Create 24 base colors
    base_cmap = colors.LinearSegmentedColormap.from_list(
        'RdYlBu_r_custom',
        list(zip(positions, colors_list)),
        N=n_trials
    )
    base_colors = base_cmap(np.linspace(0, 1, n_trials))

    # Generate 5 gradient colors for each base color via alpha variation
    gradient_colors = []
    for base_color in base_colors:
        alphas = np.linspace(0.6, 0.8, n_sub_points)
        for alpha in alphas:
            color = base_color.copy()
            color[3] = alpha
            gradient_colors.append(color)

    return colors.ListedColormap(gradient_colors)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_rearranged_patterns(feat_tsne, all_mice_data_label, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))

    arranged_data = feat_tsne
    custom_cmap = create_gradient_colormap(24, 5)

    unique_labels = np.unique(all_mice_data_label)
    for label in unique_labels:
        indices = all_mice_data_label == label
        # Color index: 5 sub-points per trial map to consecutive colormap entries
        original_trial = (label - 1) // 60   # trial index (0-23)
        sub_trial      = ((label - 1) % 60) // 12  # sub-trial index (0-4)
        color_idx = original_trial * 5 + sub_trial

        ax.scatter(arranged_data[indices, 0], arranged_data[indices, 1],
                   c=[custom_cmap.colors[color_idx]],
                   s=10.0,
                   alpha=0.8,
                   edgecolors='none')

    # Show all four spines (closed box)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')

    # Tick style
    ax.xaxis.set_tick_params(direction='out', width=0.8, length=4)
    ax.yaxis.set_tick_params(direction='out', width=0.8, length=4)

    # Show only the center tick on each axis
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_center = int(round((x_min + x_max) / 2))
    y_center = int(round((y_min + y_max) / 2))
    ax.set_xticks([x_center])
    ax.set_yticks([y_center])
    ax.set_xticklabels([f'{x_center}'])
    ax.set_yticklabels([f'{y_center}'])

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    if title:
        ax.set_title(title, fontsize=12)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    plt.tight_layout()
    return fig, ax


def rearrange_mouse_patterns(feat_tsne, all_mice_data_label):
    n_points_per_pattern = 120
    n_patterns = len(feat_tsne) // n_points_per_pattern

    patterns = []
    for i in range(n_patterns):
        start_idx = i * n_points_per_pattern
        end_idx   = start_idx + n_points_per_pattern
        pattern_data = feat_tsne[start_idx:end_idx]

        center            = np.mean(pattern_data, axis=0)
        relative_positions = pattern_data - center
        patterns.append({
            'data':               pattern_data,
            'center':             center,
            'relative_positions': relative_positions,
            'indices':            list(range(start_idx, end_idx))
        })

    spacing = 12
    positions = [
        [0,          spacing], [spacing,   spacing], [2 * spacing, spacing],
        [0,          0      ], [spacing,   0      ], [2 * spacing, 0      ]
    ]

    new_positions = feat_tsne.copy()
    for i, pattern in enumerate(patterns):
        if i >= len(positions):
            break
        new_center = np.array(positions[i])
        for idx, rel_pos in zip(pattern['indices'], pattern['relative_positions']):
            new_positions[idx] = new_center + rel_pos

    return new_positions


# ── Data loading ──────────────────────────────────────────────────────────────
def load_mouse_data(all_mouse_dict, mice_list, split_key, feature_type):
    """
    Extract and flatten latent features from a given split (train / val).

    Args:
        all_mouse_dict : top-level dict loaded from a pkl file
        mice_list      : list of mouse keys to iterate over
        split_key      : 'train' or 'val'
        feature_type   : field name inside each session dict (e.g. 'pld', 'ild', 'latent_variable')

    Returns:
        data_list  : list of 1-D numpy arrays (one per session)
        label_list : list of int labels derived from session filenames
    """
    data_list  = []
    label_list = []

    for mouse in mice_list:
        if mouse not in all_mouse_dict:
            print(f"  [Warning] Mouse '{mouse}' not found in pkl, skipping.")
            continue

        split_dict = all_mouse_dict[mouse].get(split_key, {})
        for trial_name, trial_data in split_dict.items():
            if feature_type not in trial_data:
                print(f"  [Warning] feature '{feature_type}' not found in "
                      f"{mouse}/{split_key}/{trial_name}, skipping.")
                continue

            feat = trial_data[feature_type]
            data_list.append(np.array(feat).reshape(-1))
            label_list.append(int(trial_name[:-3]))  # strip '.pt' suffix

    return data_list, label_list


for method_name, method_cfg in config['latent_methods'].items():
    pkl_path     = method_cfg['pkl_path']
    feature_type = method_cfg['feature_type']

    print(f"\n{'='*60}")
    print(f"Method : {method_name}")
    print(f"PKL    : {pkl_path}")
    print(f"Feature: {feature_type}")
    print('='*60)

    if not os.path.exists(pkl_path):
        print(f"  [Error] File not found: {pkl_path}, skipping.")
        continue

    with open(pkl_path, 'rb') as f:
        all_mouse_dict = pickle.load(f)

    # Load train split
    train_data,  train_labels  = load_mouse_data(
        all_mouse_dict, mice_train, 'train', feature_type)

    # Load val split
    val_data, val_labels = load_mouse_data(
        all_mouse_dict, mice_val, 'val', feature_type)

    all_mice_data       = np.array(train_data  + val_data)
    all_mice_data_label = np.array(train_labels + val_labels)

    print(f"  Total samples: {len(all_mice_data)}, labels: {len(all_mice_data_label)}")

    # t-SNE
    tsne      = TSNE(n_components=2, random_state=42)
    feat_tsne = tsne.fit_transform(all_mice_data)
    print(f"  t-SNE output: {feat_tsne.shape}")

    # Plot & save
    fig, _ = plot_rearranged_patterns(feat_tsne, all_mice_data_label, title=method_name)

    safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(output_dir, f'{safe_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: {save_path}")

print("\nAll methods processed.")