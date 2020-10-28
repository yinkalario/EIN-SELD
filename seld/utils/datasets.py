from pathlib import Path
import pandas as pd


class Dcase2020task3:
    """DCASE 2020 Task 3 dataset

    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.dataset_dir = dict()
        self.dataset_dir['dev'] = {
            'foa': self.root_dir.joinpath('foa_dev'),
            'mic': self.root_dir.joinpath('mic_dev'),
            'meta': self.root_dir.joinpath('metadata_dev'),            
        }
        self.dataset_dir['eval'] = {
            'foa': self.root_dir.joinpath('foa_eval'),
            'mic': self.root_dir.joinpath('mic_eval'),
            'meta': self.root_dir.joinpath('metadata_eval'),            
        }       

        self.label_set = ['alarm', 'crying baby', 'crash', 'barking dog', 'running engine', 'female scream', \
            'female speech', 'burning fire', 'footsteps', 'knocking on door', 'male scream', 'male speech', \
                'ringing phone', 'piano']
        
        self.clip_length = 60 # seconds long
        self.label_resolution = 0.1    # 0.1s is the label resolution
        self.fold_str_index = 4 # string index indicating fold number
        self.ov_str_index = -1  # string index indicating overlap
