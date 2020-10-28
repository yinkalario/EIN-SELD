import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class CheckpointIO:
    """CheckpointIO class.

    It handles saving and loading checkpoints.
    """

    def __init__(self, checkpoints_dir, model, optimizer, batch_sampler, metrics_names, num_checkpoints=1, remark=None):
        """
        Args:
            checkpoint_dir (Path obj): path where checkpoints are saved
            model: model
            optimizer: optimizer
            batch_sampler: batch_sampler
            metrics_names: metrics names to be saved in a checkpoints csv file
            num_checkpoints: maximum number of checkpoints to save. When it exceeds the number, the older 
                (older, smaller or higher) checkpoints will be deleted
            remark (optional): to remark the name of the checkpoint
        """
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.batch_sampler = batch_sampler
        self.num_checkpoints = num_checkpoints
        self.remark = remark

        self.value_list = []
        self.epoch_list = []

        self.checkpoints_csv_path = checkpoints_dir.joinpath('metrics_statistics.csv')

        # save checkpoints_csv header
        metrics_keys_list = [name for name in metrics_names]
        header = ['epoch'] + metrics_keys_list
        df_header = pd.DataFrame(columns=header)
        df_header.to_csv(self.checkpoints_csv_path, sep='\t', index=False, mode='a+')    

    def save(self, epoch, it, metrics, key_rank=None, rank_order='high'):
        """Save model. It will save a latest model, a best model of rank_order for value, and 
        'self.num_checkpoints' best models of rank_order for value.

        Args:
            metrics: metrics to log
            key_rank (str): the key of metrics to rank
            rank_order: 'low' | 'high' | 'latest'
                'low' to keep the models of lowest values
                'high' to keep the models of highest values
                'latest' to keep the models of latest epochs
        """

        ## save checkpionts_csv
        metrics_values_list = [value for value in metrics.values()]
        checkpoint_list = [[epoch] + metrics_values_list]
        df_checkpoint = pd.DataFrame(checkpoint_list)
        df_checkpoint.to_csv(self.checkpoints_csv_path, sep='\t', header=False, index=False, mode='a+')

        ## save checkpoints
        current_value = None if rank_order == 'latest' else metrics[key_rank]

        # latest model
        latest_checkpoint_path = self.checkpoints_dir.joinpath('{}_epoch_latest.pth'.format(self.remark))
        self.save_file(latest_checkpoint_path, epoch, it)

        # self.num_checkpoints best models
        if len(self.value_list) < self.num_checkpoints:
            self.value_list.append(current_value)
            self.epoch_list.append(epoch)
            checkpoint_path = self.checkpoints_dir.joinpath('{}_epoch_{}.pth'.format(self.remark, epoch))
            self.save_file(checkpoint_path, epoch, it)
            logging.info('Checkpoint saved to {}'.format(checkpoint_path))
        elif len(self.value_list) >= self.num_checkpoints:
            value_list = np.array(self.value_list)
            if rank_order == 'high' and current_value >= value_list.min():
                worst_index = value_list.argmin()
                self.del_and_save(worst_index, current_value, epoch, it)
            elif rank_order == 'low' and current_value <= value_list.max():
                worst_index = value_list.argmax()
                self.del_and_save(worst_index, current_value, epoch, it)            
            elif rank_order == 'latest':
                worst_index = 0
                self.del_and_save(worst_index, current_value, epoch, it) 

        # best model
        value_list = np.array(self.value_list)
        best_checkpoint_path = self.checkpoints_dir.joinpath('{}_epoch_best.pth'.format(self.remark))
        if rank_order == 'high' and current_value >= value_list.max():
            self.save_file(best_checkpoint_path, epoch, it)
        elif rank_order == 'low' and current_value <= value_list.min():
            self.save_file(best_checkpoint_path, epoch, it)
        elif rank_order == 'latest':
            self.save_file(best_checkpoint_path, epoch, it)

    def del_and_save(self, worst_index, current_value, epoch, it):
        """Delete and save checkpoint
        
        Args:
            worst_index: worst index,
            current_value: current value,
            epoch: epoch,
            it: it,     
        """
        worst_chpt_path = self.checkpoints_dir.joinpath('{}_epoch_{}.pth'.format(self.remark, self.epoch_list[worst_index]))
        if worst_chpt_path.is_file():
            worst_chpt_path.unlink()
        self.value_list.pop(worst_index)
        self.epoch_list.pop(worst_index)

        self.value_list.append(current_value)
        self.epoch_list.append(epoch)
        checkpoint_path = self.checkpoints_dir.joinpath('{}_epoch_{}.pth'.format(self.remark, epoch))
        self.save_file(checkpoint_path, epoch, it)
        logging.info('Checkpoint saved to {}'.format(checkpoint_path))

    def save_file(self, checkpoint_path, epoch, it):
        """Save a module to a file

        Args:
            checkpoint_path (Path obj): checkpoint path, including .pth file name
            epoch: epoch,
            it: it
        """
        outdict = {
            'epoch': epoch,
            'it': it,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'sampler': self.batch_sampler.get_state(),
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(),
            'random': random.getstate(),
            'np_random': np.random.get_state(),
        }
        torch.save(outdict, checkpoint_path)

    def load(self, checkpoint_path):
        """Load a module from a file
        
        """
        state_dict = torch.load(checkpoint_path)
        epoch = state_dict['epoch']
        it = state_dict['it']
        self.model.module.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.batch_sampler.set_state(state_dict['sampler'])
        torch.set_rng_state(state_dict['rng'])
        torch.cuda.set_rng_state(state_dict['cuda_rng'])
        random.setstate(state_dict['random'])
        np.random.set_state(state_dict['np_random'])
        logging.info('Resuming complete from {}\n'.format(checkpoint_path))
        return epoch, it

