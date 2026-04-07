import numpy as np
import scipy.io as scio
import torch
import pickle

date = '20210916'

# load data
def load_SCN(scn_data_path, task):

    scn_data = scio.loadmat(scn_data_path)
    trace = scn_data['trace'].T # trace
    poi = torch.FloatTensor(scn_data['POI'])

    if task == 'standard' or task == 'pc-sample':
        trace = trace[:,0:4800]
        train = np.reshape(trace, (trace.shape[0], 24, 200))
    elif task == 'time-sample':
        trace = trace[:,0:2400]
        train = np.reshape(trace, (trace.shape[0], 24, 100))
    elif task == '1_3-sample':
        trace = trace[:,0:1600]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '2_3-sample':
        trace = trace[:,1600:3200]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '3_3-sample':
        trace = trace[:,3200:4800]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
        
    return train, poi


def load_generated_SCN(pkl_path, mouse_key, task='standard'):
    """
    加载生成的SCN数据（pkl格式）
    
    Args:
        pkl_path: pkl文件路径，如 'generated_activity_repeat1.pkl'
        mouse_key: 小鼠标识符，如 'all_neuron_20210916'
        task: 任务类型
    
    Returns:
        train: numpy array, shape (N, num_trials, trial_length)
        poi: torch.FloatTensor, shape (N, 3)
    """
    with open(pkl_path, 'rb') as f:
        all_mice_data = pickle.load(f)
    
    data = all_mice_data[mouse_key]
    trace = data['activity']  # (N, 4800)
    poi = torch.FloatTensor(data['position'])  # (N, 3)
    
    if task == 'standard' or task == 'pc-sample':
        trace = trace[:, 0:4800]
        train = np.reshape(trace, (trace.shape[0], 24, 200))
    elif task == 'time-sample':
        trace = trace[:, 0:2400]
        train = np.reshape(trace, (trace.shape[0], 24, 100))
    elif task == '1_3-sample':
        trace = trace[:, 0:1600]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '2_3-sample':
        trace = trace[:, 1600:3200]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '3_3-sample':
        trace = trace[:, 3200:4800]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    
    return train, poi


def get_available_mice(pkl_path):
    """
    获取pkl文件中所有可用的小鼠标识符
    
    Args:
        pkl_path: pkl文件路径
    
    Returns:
        list: 小鼠标识符列表
    """
    with open(pkl_path, 'rb') as f:
        all_mice_data = pickle.load(f)
    return list(all_mice_data.keys())