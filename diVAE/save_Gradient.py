import logging
from argparse import ArgumentParser
import torch
from torch.nn.parallel import DataParallel
from mmcv import Config
import random
import numpy as np
from utils import scn_loader_val, ValDataset
from models import make_diVAE
import warnings
import pickle
warnings.filterwarnings("ignore")

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
    return parser.parse_args()

def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_batch(model, batch, mouse, frame, mode='sild'):
    cal_signal = batch['data'].float()  # [6049, 40]
    cal_label = batch['label']
    
    with torch.autograd.set_grad_enabled(True):
        segments = [
            cal_signal[:, 0:16], 
            cal_signal[:, 16:32],  
            cal_signal[:, -16:] 
        ]
        stacked_cal_signal = torch.stack(segments, dim=0)  # [3, 6049, 16]
        
        stacked_cal_signal = stacked_cal_signal.cuda()
        cal_label = torch.tensor(cal_label).cuda()
        cal_label = cal_label.expand(stacked_cal_signal.shape[0])
        
        stacked_cal_signal.requires_grad = True
        stacked_cal_signal.retain_grad()
        
        pld_mean, ild_mean = model(cal_signal=stacked_cal_signal,
                                mice=mouse,
                                time_label=cal_label)
        
        if mode == 'sild':
            grads = torch.autograd.grad(torch.cat([pld_mean,ild_mean],dim=1).sum(), stacked_cal_signal)[0]
        elif mode == 'pld':
            grads = torch.autograd.grad(pld_mean.sum(), stacked_cal_signal)[0]
        else:  # ild
            grads = torch.autograd.grad(ild_mean.sum(), stacked_cal_signal)[0]
        
        full_input_weights = torch.cat([
            grads[0].detach(), 
            grads[1].detach(), 
            grads[2, :, -8:].detach(), 
        ], dim=1)  
        full_cal_signal = torch.cat([
            stacked_cal_signal[0].detach(),  
            stacked_cal_signal[1].detach(), 
            stacked_cal_signal[2, :, -8:].detach() 
        ], dim=1) 
        
        return {
            'grad': full_input_weights.abs().cpu().numpy(),
            'ca': full_cal_signal.cpu().numpy(),
            'pos': batch['pos'] if 'pos' in batch else None
        }

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    set_random_seed(cfg.seed)
    logger = get_logger(cfg.log_level)
    frame = 16  
    
    mice = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']

    train_loaders = scn_loader_val(mice=mice, batch_size=1)
    model = make_DLD_Ca(dataloaders=train_loaders,
                     proj_dict=cfg.model_dict['proj_dict'],
                     encoder_decoder_dict=cfg.model_dict['encoder_decoder_dict'],
                     pld_dict=cfg.model_dict['pld_dict'],
                     ild_dict=cfg.model_dict['ild_dict'],
                     grid_dict=cfg.model_dict['grid_dict'])
    
    model.load_state_dict(torch.load(cfg.load_from)['state_dict'], strict=False)
    model = DataParallel(model, device_ids=cfg.gpus).cuda()
    model.eval()

    all_mouse_dict = {mouse: {'sild': {}, 'pld': {}, 'ild': {}} for mouse in mice}

    for mouse in mice:
        print('processing:', mouse)
        dataset = ValDataset(mouse, frames=frame)
        
        for idx in range(len(dataset)):
            batch = dataset[idx]
            batch['pos'] = dataset.position
            
            for mode in ['sild', 'pld', 'ild']:
                results = process_batch(model, batch, mouse, frame, mode)
                all_mouse_dict[mouse][mode][f'{idx}.pt'] = results

    with open(f'./middle_state/all_lv_grad.pkl', "wb") as tf:
        pickle.dump(all_mouse_dict, tf)

if __name__ == '__main__':
    main()