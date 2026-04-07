import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from einops import rearrange
from mmcv import Config
from mmcv.runner import Runner

from models import make_diVAE
from utils import scn_loader_train, scn_loader_val, CycleDataloaders
import random
import numpy as np
import warnings
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

    set_random_seed(cfg.seed)

    logger = get_logger(cfg.log_level)
    
    logger.info('Disabled distributed training.')

    
    num_workers = cfg.data_workers * len(cfg.gpus)
    batch_size = cfg.batch_size
    
    
    mice = ['SCN1','SCN2','SCN3','SCN4','SCN5','SCN6']
    all_loaders = scn_loader_train(mice = mice, 
                            batch_size = batch_size,
                            frames=cfg.model_dict['proj_dict']['frames'])
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
    train_loaders = scn_loader_train(mice = mice_train,
                            batch_size = batch_size,
                            frames=cfg.model_dict['proj_dict']['frames'])
    
    mice_val=['SCN5','SCN6']
    val_loaders = scn_loader_val(mice = mice_val, 
                                batch_size = 1,
                                frames=cfg.model_dict['proj_dict']['frames'])
    
    model = DataParallel(model, device_ids=cfg.gpus).cuda()
    
    logger.info(f"lr: {cfg.optimizer['lr']}, " \
                f"epoch_size: {len(CycleDataloaders(train_loaders['train']))}, " \
                f"total_epochs: {cfg.total_epochs}, " \
                f"total_iters: {cfg.total_epochs*len(CycleDataloaders(train_loaders['train']))}, "
            )
    kl_coeff = cfg.coeffKL

    def batch_processor(model, data, train_mode, cur_iters):
        data_key, real_data = data
        cal_data =  real_data['data']
        cal_label =  real_data['label']
        cal_data = cal_data.cuda(non_blocking=True)
        cal_label = cal_label.cuda(non_blocking=True)
        
        if train_mode:
            pass
        else:
            cal_data = rearrange(cal_data, 'b n m t -> (b n) m t')
            cal_label = cal_label.expand(cal_data.shape[0])
        pred_ca, codebook_loss, kl_loss = model(cal_signal=cal_data, 
                                                mice=data_key,
                                                time_label=cal_label)
        loss_ca = F.mse_loss(pred_ca, cal_data, reduction='sum')
        total_loss = (loss_ca + codebook_loss.sum() + kl_coeff*kl_loss.sum())/cal_data.shape[0]
        mse = F.mse_loss(pred_ca, cal_data, reduction='mean')
        
        log_vars = OrderedDict()
        log_vars['ca_mse'] = loss_ca.item()/cal_data.shape[0]
        log_vars['pld_codebook_loss'] = codebook_loss.sum().item()/cal_data.shape[0]
        log_vars['ild_kl_loss'] = kl_loss.sum().item()/cal_data.shape[0]
        log_vars['mse'] = mse.item()
        log_vars['whole_loss'] = total_loss.item()
        
        outputs = dict(loss=total_loss, log_vars=log_vars, num_samples=cal_data.shape[0])
        return outputs

    # build runner and register hooks
    
    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        log_level=cfg.log_level)
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)
    
    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run([CycleDataloaders(train_loaders['train']), CycleDataloaders(val_loaders['val'])], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
