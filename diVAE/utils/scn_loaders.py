
import torch.utils.data as Data
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader
import torch
import typing

class TrainDataset(Data.Dataset):
    def __init__(self, mouse='all_data_20210916', frames=16):
        super(TrainDataset, self).__init__()
        with open(os.path.join('../../../data/all_scn.pkl'), "rb") as tf:
            all_scn_dict = pickle.load(tf)
        
        original_data = all_scn_dict[mouse]['data']  # 6049 * 24 * 200
        self.position = all_scn_dict[mouse]['position']
        self.neurons = all_scn_dict[mouse]['position'].shape[0]
        self.frames = frames

        expanded_data = []
        for trial in range(24):
            trial_data = original_data[:, trial, :]  # 6049 * 200
            for sub_trial in range(5):
                start_idx = sub_trial * (200 // 5)
                end_idx = start_idx + (200 // 5)
                sub_data = trial_data[:, start_idx:end_idx]  # 6049 * 40
                expanded_data.append(sub_data)
        
        self.data = torch.from_numpy(np.stack(expanded_data, axis=1))  # 6049 * 120 * 40
        
        self.labels = [(trial//5)*60 + (trial%5) + 1 for trial in range(120)]

    def __getitem__(self, index):
        all_neural_data = self.data[:, index, :]  # 6049 * 40
        max_start = all_neural_data.shape[1] - self.frames
        rand_index = np.random.randint(0, high=max_start+1)
        sub_data = all_neural_data[:, rand_index:rand_index+self.frames]
        
        out = {}
        out['data'] = sub_data
        out['label'] = self.labels[index]
        return out

    def __len__(self):
        return 120  # 24 * 5

class ValDataset(Data.Dataset):
    def __init__(self, mouse='all_data_20210916', frames=16):
        super(ValDataset, self).__init__()
        with open(os.path.join('../../../data/all_scn.pkl'), "rb") as tf:
            all_scn_dict = pickle.load(tf)
            
        original_data = all_scn_dict[mouse]['data']  # 6049 * 24 * 200
        self.position = all_scn_dict[mouse]['position']
        self.neurons = all_scn_dict[mouse]['position'].shape[0]
        self.frames = frames

        expanded_data = []
        for trial in range(24):
            trial_data = original_data[:, trial, :]  # 6049 * 200
            for sub_trial in range(5):
                start_idx = sub_trial * (200 // 5)
                end_idx = start_idx + (200 // 5)
                sub_data = trial_data[:, start_idx:end_idx]  # 6049 * 40
                expanded_data.append(sub_data)
        
        self.data = torch.from_numpy(np.stack(expanded_data, axis=1))  # 6049 * 120 * 40
        
        self.labels = [(trial//5)*60 + (trial%5) + 1 for trial in range(120)]

    def __getitem__(self, index):
        all_neural_data = self.data[:, index, :]  # 6049 * 40
        
        if self.frames == 40:  
            stacked_frames = all_neural_data
        else:
            stacked_frames = [all_neural_data[:, i:i+self.frames] 
                            for i in range(0, 40-self.frames+1, self.frames)]
            if len(stacked_frames) == 0:
                stacked_frames = [all_neural_data[:, -self.frames:]]
            stacked_frames = torch.stack(stacked_frames, dim=0)
        
        out = {}
        out['data'] = stacked_frames
        out['label'] = self.labels[index]
        return out

    def __len__(self):
        return 120  # 24 * 5
          
class CycleDataloaders:

    def __init__(self, ds: typing.Dict[str, DataLoader]):
        self.ds = ds
        self.max_iterations = max([len(ds) for ds in self.ds.values()])

    @staticmethod
    def cycle(iterable: typing.Iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def __iter__(self):
        cycles = [self.cycle(loader) for loader in self.ds.values()]
        for mouse_id, mouse_ds, _ in zip(
            self.cycle(self.ds.keys()),
            (self.cycle(cycles)),
            range(len(self.ds) * self.max_iterations),
        ):
            yield mouse_id, next(mouse_ds)

    def __len__(self):
        return len(self.ds) * self.max_iterations

def scn_loader_train(
    mice,
    batch_size=32,
    frames=32
):
    dataloaders_combined = {}
    for mosue_key in mice:
        
        dataset = TrainDataset(mosue_key,frames=frames)
        dataloaders = {}
        dataloaders['train'] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=4
        )

        dataset_name = mosue_key
        for k, v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][dataset_name] = v
    return dataloaders_combined

def scn_loader_val(
    mice,
    batch_size=32,
    frames=32
):
    dataloaders_combined = {}
    for mosue_key in mice:
        dataset = ValDataset(mosue_key,
                             frames=frames)
        dataloaders = {}
        dataloaders['val'] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=4
        )
        dataset_name = mosue_key
        for k, v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][dataset_name] = v
    return dataloaders_combined