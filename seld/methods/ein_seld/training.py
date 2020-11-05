import random
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import torch
from methods.training import BaseTrainer
from methods.utils.data_utilities import to_metrics2020_format


class Trainer(BaseTrainer):

    def __init__(self, args, cfg, dataset, af_extractor, valid_set, model, optimizer, losses, metrics):

        super().__init__()
        self.cfg = cfg
        self.af_extractor = af_extractor
        self.model = model
        self.optimizer = optimizer
        self.losses = losses
        self.metrics = metrics
        self.cuda = args.cuda

        self.clip_length = dataset.clip_length
        self.label_resolution = dataset.label_resolution
        self.label_interp_ratio = int(self.label_resolution * cfg['data']['sample_rate'] / cfg['data']['hop_length'])

        # Load ground truth for dcase metrics
        self.num_segments = valid_set.num_segments
        self.valid_gt_sed_metrics2019 = valid_set.valid_gt_sed_metrics2019
        self.valid_gt_doa_metrics2019 = valid_set.valid_gt_doa_metrics2019
        self.gt_metrics2020_dict = valid_set.gt_metrics2020_dict

        # Scalar
        scalar_h5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('scalar')
        fn_scalar = '{}_{}_sr{}_nfft{}_hop{}_mel{}.h5'.format(cfg['data']['type'], cfg['data']['audio_feature'], 
            cfg['data']['sample_rate'], cfg['data']['n_fft'], cfg['data']['hop_length'], cfg['data']['n_mels'])
        scalar_path = scalar_h5_dir.joinpath(fn_scalar)
        with h5py.File(scalar_path, 'r') as hf:
            self.mean = hf['mean'][:]
            self.std = hf['std'][:]
        if args.cuda:
            self.mean = torch.tensor(self.mean, dtype=torch.float32).cuda()
            self.std = torch.tensor(self.std, dtype=torch.float32).cuda()

        self.init_train_losses()
    
    def init_train_losses(self):
        """ Initialize train losses

        """
        self.train_losses = {
            'loss_all': 0.,
            'loss_sed': 0.,
            'loss_doa': 0.
        }

    def train_step(self, batch_sample, epoch_it):
        """ Perform a train step

        """
        batch_x = batch_sample['waveform']
        batch_target = {
            'ov': batch_sample['ov'],
            'sed': batch_sample['sed_label'],
            'doa': batch_sample['doa_label']
        }
        if self.cuda:
            batch_x = batch_x.cuda(non_blocking=True)
            batch_target['sed'] = batch_target['sed'].cuda(non_blocking=True)
            batch_target['doa'] = batch_target['doa'].cuda(non_blocking=True)

        self.optimizer.zero_grad()
        self.af_extractor.train()
        self.model.train()
        batch_x = self.af_extractor(batch_x)
        batch_x = (batch_x - self.mean) / self.std
        pred = self.model(batch_x)
        loss_dict = self.losses.calculate(pred, batch_target)
        loss_dict[self.cfg['training']['loss_type']].backward()
        self.optimizer.step()

        self.train_losses['loss_all'] += loss_dict['all']
        self.train_losses['loss_sed'] += loss_dict['sed']
        self.train_losses['loss_doa'] += loss_dict['doa']
        

    def validate_step(self, generator=None, max_batch_num=None, valid_type='train', epoch_it=0):
        """ Perform the validation on the train, valid set

        Generate a batch of segmentations each time
        """

        if valid_type == 'train':
            train_losses = self.train_losses.copy()
            self.init_train_losses()
            return train_losses

        elif valid_type == 'valid':
            pred_sed_list, pred_doa_list = [], []
            gt_sed_list, gt_doa_list = [], []
            loss_all, loss_sed, loss_doa = 0., 0., 0.

            for batch_idx, batch_sample in enumerate(generator):
                if batch_idx == max_batch_num:
                    break

                batch_x = batch_sample['waveform']
                batch_target = {
                    'sed': batch_sample['sed_label'],
                    'doa': batch_sample['doa_label']
                }

                if self.cuda:
                    batch_x = batch_x.cuda(non_blocking=True)
                    batch_target['sed'] = batch_target['sed'].cuda(non_blocking=True)
                    batch_target['doa'] = batch_target['doa'].cuda(non_blocking=True)

                with torch.no_grad():
                    self.af_extractor.eval()
                    self.model.eval()
                    batch_x = self.af_extractor(batch_x)
                    batch_x = (batch_x - self.mean) / self.std
                    pred = self.model(batch_x)
                loss_dict = self.losses.calculate(pred, batch_target, epoch_it)
                pred['sed'] = torch.sigmoid(pred['sed'])
                loss_all += loss_dict['all'].cpu().detach().numpy()
                loss_sed += loss_dict['sed'].cpu().detach().numpy()
                loss_doa += loss_dict['doa'].cpu().detach().numpy()
                pred_sed_list.append(pred['sed'].cpu().detach().numpy())
                pred_doa_list.append(pred['doa'].cpu().detach().numpy())

            pred_sed = np.concatenate(pred_sed_list, axis=0)
            pred_doa = np.concatenate(pred_doa_list, axis=0)

            origin_num_clips = int(pred_sed.shape[0]/self.num_segments)
            origin_T = int(pred_sed.shape[1]*self.num_segments)
            pred_sed = pred_sed.reshape((origin_num_clips, origin_T, 2, -1))[:, :int(self.clip_length / self.label_resolution)]
            pred_doa = pred_doa.reshape((origin_num_clips, origin_T, 2, -1))[:, :int(self.clip_length / self.label_resolution)]

            pred_sed_max = pred_sed.max(axis=-1)
            pred_sed_max_idx = pred_sed.argmax(axis=-1)
            pred_sed = np.zeros_like(pred_sed)
            for b_idx in range(origin_num_clips):
                for t_idx in range(origin_T):
                    for track_idx in range(2):
                        pred_sed[b_idx, t_idx, track_idx, pred_sed_max_idx[b_idx, t_idx, track_idx]] = \
                            pred_sed_max[b_idx, t_idx, track_idx]
            pred_sed = (pred_sed > self.cfg['training']['threshold_sed']).astype(np.float32)
            
            # convert Catesian to Spherical
            azi = np.arctan2(pred_doa[..., 1], pred_doa[..., 0])
            elev = np.arctan2(pred_doa[..., 2], np.sqrt(pred_doa[..., 0]**2 + pred_doa[..., 1]**2))
            pred_doa = np.stack((azi, elev), axis=-1) # (N, T, tracks, (azi, elev))

            # convert format
            pred_sed_metrics2019, pred_doa_metrics2019 = to_metrics2019_format(pred_sed, pred_doa)
            gt_sed_metrics2019, gt_doa_metrics2019 = self.valid_gt_sed_metrics2019, self.valid_gt_doa_metrics2019
            pred_dcase_format_dict = to_dcase_format(pred_sed, pred_doa)
            pred_metrics2020_dict = to_metrics2020_format(pred_dcase_format_dict, 
                pred_sed.shape[0]*pred_sed.shape[1], label_resolution=self.label_resolution)
            gt_metrics2020_dict = self.gt_metrics2020_dict

            out_losses = {
                'loss_all': loss_all / (batch_idx + 1),
                'loss_sed': loss_sed / (batch_idx + 1),
                'loss_doa': loss_doa / (batch_idx + 1),
            }

            pred_dict = {
                'dcase2019_sed': pred_sed_metrics2019,
                'dcase2019_doa': pred_doa_metrics2019,
                'dcase2020': pred_metrics2020_dict,
            }

            gt_dict = {
                'dcase2019_sed': gt_sed_metrics2019,
                'dcase2019_doa': gt_doa_metrics2019,
                'dcase2020': gt_metrics2020_dict,
            }
            metrics_scores = self.metrics.calculate(pred_dict, gt_dict)
            return out_losses, metrics_scores


def to_metrics2019_format(sed_labels, doa_labels):
    """Convert sed and doa labels from track-wise output format to DCASE2019 evaluation metrics input format

    Args:
        sed_labels: SED labels, (batch_size, time_steps, num_tracks=2, logits_events=14 (number of classes))
        doa_labels: DOA labels, (batch_size, time_steps, num_tracks=2, logits_degrees=2 (azi in radians, ele in radians))
    Output:
        out_sed_labels: SED labels, (batch_size * time_steps, logits_events=14 (True or False)
        out_doa_labels: DOA labels, (batch_size * time_steps, azi_index=14 + ele_index=14)
    """
    batch_size, T, num_tracks, num_classes = sed_labels.shape
    sed_labels = sed_labels.reshape(batch_size * T, num_tracks, num_classes)
    doa_labels = doa_labels.reshape(batch_size * T, num_tracks, 2)
    out_sed_labels = np.logical_or(sed_labels[:, 0], sed_labels[:, 1]).astype(float)
    out_doa_labels = np.zeros((batch_size * T, num_classes * 2))
    for n_track in range(num_tracks):
        indexes = np.where(sed_labels[:, n_track, :])
        out_doa_labels[:, 0: num_classes][indexes[0], indexes[1]] = \
            doa_labels[indexes[0], n_track, 0]  # azimuth
        out_doa_labels[:, num_classes: 2*num_classes][indexes[0], indexes[1]] = \
            doa_labels[indexes[0], n_track, 1]  # elevation
    return out_sed_labels, out_doa_labels

def to_dcase_format(sed_labels, doa_labels):
    """Convert sed and doa labels from track-wise output format to dcase output format

    Args:
        sed_labels: SED labels, (batch_size, time_steps, num_tracks=2, logits_events=14 (number of classes))
        doa_labels: DOA labels, (batch_size, time_steps, num_tracks=2, logits_degrees=2 (azi in radiance, ele in radiance))
    Output:
        output_dict: return a dict containing dcase output format
            output_dict[frame-containing-events] = [[class_index_1, azi_1 in degree, ele_1 in degree], [class_index_2, azi_2 in degree, ele_2 in degree]]
    """
    batch_size, T, num_tracks, num_classes= sed_labels.shape

    sed_labels = sed_labels.reshape(batch_size*T, num_tracks, num_classes)
    doa_labels = doa_labels.reshape(batch_size*T, num_tracks, 2)
    
    output_dict = {}
    for n_idx in range(batch_size*T):
        for n_track in range(num_tracks):
            class_index = list(np.where(sed_labels[n_idx, n_track, :])[0])
            assert len(class_index) <= 1, 'class_index should be smaller or equal to 1!!\n'
            if class_index:
                event_doa = [class_index[0], int(np.around(doa_labels[n_idx, n_track, 0] * 180 / np.pi)), \
                                            int(np.around(doa_labels[n_idx, n_track, 1] * 180 / np.pi))] # NOTE: this is in degree
                if n_idx not in output_dict:
                    output_dict[n_idx] = []
                output_dict[n_idx].append(event_doa)
    return output_dict

