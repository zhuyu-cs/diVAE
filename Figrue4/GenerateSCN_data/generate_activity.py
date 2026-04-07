import logging
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DataParallel
from einops import rearrange
import os
from models import make_diVAE
from utils import scn_loader_val, ValDataset
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore") 
import pickle
import math


def get_logger(log_level):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger


def parse_args():
    parser = ArgumentParser(description='Generate neural activity from priors')
    parser.add_argument('--num_repeats', type=int, default=5, 
                        help='number of sampling repeats')
    parser.add_argument('--output_dir', type=str, default='./generated_activity/',
                        help='output directory for generated pkl files')
    parser.add_argument('--step', type=int, default=1,
                        help='step size for sliding window (1 = most smooth)')
    parser.add_argument('--blend_mode', type=str, default='avg',
                        choices=['avg', 'weighted_avg', 'center'],
                        help='blending mode for overlapping regions')
    return parser.parse_args()


def set_random_seed(seed, deterministic=True):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings for timesteps.
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def generate_from_priors(model, time_label, mice, num_frames, device):
    """
    Generate neural activity by sampling from ILD and PLD priors.
    
    Args:
        model: the DistinctLatentDynamics model
        time_label: tensor of shape (B,) with time labels
        mice: mouse identifier string  
        num_frames: number of time frames in the output
        device: torch device
    
    Returns:
        generated_activity: tensor of shape (B, N_neurons, num_frames)
    """
    if isinstance(model, DataParallel):
        model_inner = model.module
    else:
        model_inner = model
    
    B = time_label.shape[0]
    
    num_resolutions = len(model_inner.encoder.down)
    spatial_dim = model_inner.in_projectors.proj_neuron // (2 ** (num_resolutions - 1))
    
    # ===== Get ILD prior from position coordinates =====
    positions = model_inner.ilds.grids_per_mice[mice].unsqueeze(0).permute(0, 2, 1).to(device)
    
    uc_mean, uc_log_var = model_inner.ilds.grid_pridictor(positions)
    
    uc_mean = uc_mean.unsqueeze(-1).expand(B, -1, -1, num_frames)
    uc_log_var = uc_log_var.unsqueeze(-1).expand(B, -1, -1, num_frames)
    
    # Sample from ILD prior
    ild_sample = uc_mean + torch.exp(0.5 * uc_log_var) * torch.randn_like(uc_mean)
    
    # ===== Get PLD prior from time label =====
    embedding_dim = model_inner.pld.embedding_dim
    time_embed = get_timestep_embedding(time_label, embedding_dim * 8)
    label_mapped = model_inner.pld.label_mapping(time_embed).unsqueeze(-1)
    
    ut_mean = model_inner.pld.u_mean(label_mapped)
    ut_log_var = model_inner.pld.u_logvar(label_mapped)
    
    ut_mean = ut_mean.unsqueeze(-1).expand(B, -1, spatial_dim, num_frames)
    ut_log_var = ut_log_var.unsqueeze(-1).expand(B, -1, spatial_dim, num_frames)
    
    # Sample from PLD prior
    pld_sample = ut_mean + torch.exp(0.5 * ut_log_var) * torch.randn_like(ut_mean)
    
    # ===== Concatenate and decode =====
    latent = torch.cat([ild_sample, pld_sample], dim=1)
    decoded = model_inner.decoder(latent)
    output = model_inner.output_projectors(decoded, mice)
    
    return output


def generate_with_step1(model, label, mouse, total_frames, gen_frames,
                        output_frames, device, step=1, blend_mode='avg'):
    time_label_tensor = torch.tensor([label]).to(device)
    
    if output_frames >= total_frames:
        generated = generate_from_priors(model, time_label_tensor, mouse, gen_frames, device)
        return generated.squeeze(0).cpu().numpy()[:, :total_frames]
    
    num_windows = total_frames - output_frames + 1
    
    windows = []
    for window_idx in range(0, num_windows, step):
        generated = generate_from_priors(model, time_label_tensor, mouse, gen_frames, device)
        windows.append(generated.squeeze(0).cpu().numpy())  # (N_neurons, output_frames)
    
    n_neurons = windows[0].shape[0]
    actual_window_starts = list(range(0, num_windows, step))
    
    if blend_mode == 'center':
        result = np.zeros((n_neurons, total_frames), dtype=np.float32)
        assigned = np.zeros(total_frames, dtype=bool)
        
        center_offset = output_frames // 2
        for t in range(total_frames):
            best_window_start = t - center_offset
            best_window_start = max(0, min(best_window_start, num_windows - 1))
            
            closest_idx = np.argmin([abs(s - best_window_start) for s in actual_window_starts])
            window_start = actual_window_starts[closest_idx]
            
            offset_in_window = t - window_start
            if 0 <= offset_in_window < output_frames:
                result[:, t] = windows[closest_idx][:, offset_in_window]
                assigned[t] = True
        
        if not np.all(assigned):
            for t in np.where(~assigned)[0]:
                for idx, start in enumerate(actual_window_starts):
                    offset = t - start
                    if 0 <= offset < output_frames:
                        result[:, t] = windows[idx][:, offset]
                        break
        
        return result
    
    elif blend_mode == 'weighted_avg':
        result = np.zeros((n_neurons, total_frames), dtype=np.float32)
        weights = np.zeros(total_frames, dtype=np.float32)
        
        window_weights = np.zeros(output_frames, dtype=np.float32)
        center = output_frames / 2
        for i in range(output_frames):
            window_weights[i] = 1.0 - abs(i - center + 0.5) / center
        
        for idx, start in enumerate(actual_window_starts):
            window = windows[idx]
            for offset in range(output_frames):
                t = start + offset
                if t < total_frames:
                    w = window_weights[offset]
                    result[:, t] += window[:, offset] * w
                    weights[t] += w
        
        return result / np.maximum(weights, 1e-8)
    
    else:  
        result = np.zeros((n_neurons, total_frames), dtype=np.float32)
        counts = np.zeros(total_frames, dtype=np.float32)
        
        for idx, start in enumerate(actual_window_starts):
            window = windows[idx]
            for offset in range(output_frames):
                t = start + offset
                if t < total_frames:
                    result[:, t] += window[:, offset]
                    counts[t] += 1
        
        return result / np.maximum(counts, 1)


def generate_full_activity(model, mouse, dataset, device, gen_frames=2, 
                           output_frames=16, step=1, blend_mode='avg'):
    n_neurons = dataset.neurons
    all_subtrial_activities = []
    
    for subtrial_idx in range(120):
        label = dataset.labels[subtrial_idx]
        
        subtrial_activity = generate_with_step1(
            model=model,
            label=label,
            mouse=mouse,
            total_frames=40,
            gen_frames=gen_frames,
            output_frames=output_frames,
            device=device,
            step=step,
            blend_mode=blend_mode
        )
        
        all_subtrial_activities.append(subtrial_activity)
    
    full_activity = np.zeros((n_neurons, 4800), dtype=np.float32)
    
    for subtrial_idx in range(120):
        trial_idx = subtrial_idx // 5
        sub_idx = subtrial_idx % 5
        start_in_full = trial_idx * 200 + sub_idx * 40
        full_activity[:, start_in_full:start_in_full + 40] = all_subtrial_activities[subtrial_idx]
    
    return full_activity


def main():
    args = parse_args()
    
    logger = get_logger(logging.INFO)
    logger.info('Generating neural activity from ILD and PLD priors.')
    logger.info(f'Settings: step={args.step}, blend_mode={args.blend_mode}')
    
    # All mice
    all_mice = [
        'SCN1', 
        'SCN2', 
        'SCN3', 
        'SCN4',
    ]
    
    frame = 2  
    output_frames = 16  
    
    # Load dataloaders to initialize model
    all_loaders = scn_loader_val(mice=all_mice, batch_size=1, frames=frame)
    
    model_dict = dict(  
        proj_dict=dict(
            in_dim=1,
            proj_neuron=256,
            frames=frame,
            num_param=4,
            dropout=0.1,
            base_channel=64,
        ),
        encoder_decoder_dict=dict(  
            base_channel=32, 
            ch_mult=(1,2,2,4), 
            num_res_blocks=1,      
            attn_resolutions=[],    
            dropout=0.1, 
            resamp_with_conv=True,
            double_z=False,
            give_pre_end=False,
            z_channels=2,             
        ),
        pld_dict=dict(
            n_e=256,                   
            code_dim=2,               
        ),
        ild_dict=dict(
            latent_channel=2,          
            dropout=0.1
        ),
        grid_dict=dict(
            input_channels=3,
            hidden_channel=2,
            output_channel=2,
        )
    )

    # Build model
    model = make_diVAE(
        dataloaders=all_loaders['val'],
        proj_dict=model_dict['proj_dict'],  
        encoder_decoder_dict=model_dict['encoder_decoder_dict'],  
        pld_dict=model_dict['pld_dict'],  
        ild_dict=model_dict['ild_dict'],  
        grid_dict=model_dict['grid_dict'],
    )
    
    # Load weights
    weight_path = './pretrained_weights/weights.pth'
    
    if not os.path.exists(weight_path):
        logger.error(f'Weight file not found: {weight_path}')
        return
    
    logger.info(f'Loading weights from: {weight_path}')
    model.load_state_dict(torch.load(weight_path)['state_dict'], strict=False)
    model = DataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    device = torch.device('cuda')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    num_repeats = args.num_repeats
    logger.info(f'Will generate {num_repeats} repeats for each mouse.')
    
    total_frames_per_subtrial = 40
    num_windows_per_subtrial = (total_frames_per_subtrial - output_frames) // args.step + 1
    logger.info(f'Each subtrial ({total_frames_per_subtrial} frames) needs {num_windows_per_subtrial} windows')
    
    # ===== Generate activity for each repeat =====
    for repeat_idx in range(num_repeats):
        set_random_seed(42 + repeat_idx + 1)
        
        logger.info(f'===== Sampling repeat {repeat_idx + 1}/{num_repeats} =====')
        
        all_mouse_dict = {}
        
        for mouse in all_mice:
            logger.info(f'Processing mouse: {mouse}')
            
            dataset = ValDataset(mouse, frames=frame)
            
            if isinstance(model, DataParallel):
                positions = model.module.ilds.grids_per_mice[mouse].cpu().numpy()
            else:
                positions = model.ilds.grids_per_mice[mouse].cpu().numpy()
            
            with torch.no_grad():
                full_activity = generate_full_activity(
                    model=model,
                    mouse=mouse,
                    dataset=dataset,
                    device=device,
                    gen_frames=frame,
                    output_frames=output_frames,
                    step=args.step,
                    blend_mode=args.blend_mode
                )
            
            logger.info(f'  Generated activity shape: {full_activity.shape}')
            
            all_mouse_dict[mouse] = {
                'activity': full_activity,
                'position': positions
            }
        
        output_path = os.path.join(args.output_dir, f'generated_activity_repeat{repeat_idx + 1}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(all_mouse_dict, f)
        logger.info(f'Saved: {output_path}')
    
    logger.info('Generation complete!')


if __name__ == '__main__':
    main()