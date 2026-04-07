import logging
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DataParallel
from einops import rearrange
from mmcv import Config
import os
import glob
from models import make_diVAE
from utils import scn_loader_val, ValDataset
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore") 
import pickle
import shutil
import copy

def get_logger(log_level):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger

def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--ckpt', type=str)
    return parser.parse_args()

def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    frame = cfg.model_dict['proj_dict']['frames']
    set_random_seed(cfg.seed)

    logger = get_logger(cfg.log_level)
    
    logger.info('Disabled distributed training.')
    
    mice = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']
    all_loaders = scn_loader_val(mice = mice, 
                            batch_size = 1)
    # build model   
    model = make_diVAE(dataloaders=all_loaders,
                        proj_dict=cfg.model_dict['proj_dict'],  
                        encoder_decoder_dict=cfg.model_dict['encoder_decoder_dict'],  
                        pld_dict=cfg.model_dict['pld_dict'],  
                        ild_dict=cfg.model_dict['ild_dict'],  
                        grid_dict=cfg.model_dict['grid_dict'],
                        )
    
    del all_loaders
    mice_train = ['SCN1','SCN2','SCN3','SCN4']
    
    mice_val=['SCN5','SCN6']
    
    weight_path = cfg.load_from
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path)['state_dict'], strict=False)
        
        model = DataParallel(model, device_ids=cfg.gpus).cuda()
        
        model.eval()
        all_mouse_dict = {}
        for mouse in mice_train:
            print('Processing train mouse:', mouse)
            all_mouse_dict[mouse] = {'train': {}}
            
            dataset = ValDataset(mouse)
            
            for idx in range(len(dataset)):
                batch = dataset.__getitem__(idx)
                cal_signal = batch['data'].unsqueeze(0)  # [1, neurons, time_bin_size]
                cal_label = batch['label'] 
                
                with torch.no_grad():
                    stacked_cal_signal = cal_signal.cuda()
                    cal_label_tensor = torch.tensor(cal_label).cuda().unsqueeze(0)
                    
                    dilv_mean = model.inference(
                        cal_signal=stacked_cal_signal,
                        mice=mouse,
                        time_label=cal_label_tensor
                    )
                    dilv_mean = rearrange(dilv_mean.permute(0,3,1,2), 'b t c n -> (b t) (c n)')
                    saved_info = {
                        'latent_variable': dilv_mean.detach().cpu().numpy()
                    }
                    
                    all_mouse_dict[mouse]['train'][f'{cal_label}.pt'] = saved_info

        for mouse in mice_val:
            print('Processing val mouse:', mouse)
            all_mouse_dict[mouse] = {'val': {}}
            
            dataset = ValDataset(mouse)
            
            for idx in range(len(dataset)):
                batch = dataset.__getitem__(idx)
                cal_signal = batch['data'].unsqueeze(0)
                cal_label = batch['label']
                
                with torch.no_grad():
                    stacked_cal_signal = cal_signal.cuda()
                    cal_label_tensor = torch.tensor(cal_label).cuda().unsqueeze(0)
                    
                    dilv_mean= model.inference(
                        cal_signal=stacked_cal_signal,
                        mice=mouse,
                        time_label=cal_label_tensor
                    )
                    
                    dilv_mean = rearrange(dilv_mean.permute(0,3,1,2), 'b t c n -> (b t) (c n)')
                    
                    saved_info = {
                        'latent_variable': dilv_mean.detach().cpu().numpy()
                    }
                    all_mouse_dict[mouse]['val'][f'{cal_label}.pt'] = saved_info

        os.makedirs('./middle_state/', exist_ok=True)
        proj = cfg.model_dict['proj_dict']['proj_neuron']
        
        with open(f'./middle_state/all_lv{proj}_f{frame}.pkl', "wb") as tf:
            pickle.dump(all_mouse_dict, tf)

if __name__ == '__main__':
    main()