from pathlib import Path

import h5py
import numpy as np
import torch
from methods.inference import BaseInferer
from tqdm import tqdm

from .training import to_dcase_format


class Inferer(BaseInferer):

    def __init__(self, cfg, dataset, af_extractor, model, cuda):
        super().__init__()
        self.cfg = cfg
        self.af_extractor = af_extractor
        self.model = model
        self.cuda = cuda

        # Scalar
        scalar_h5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('scalar')
        fn_scalar = '{}_{}_sr{}_nfft{}_hop{}_mel{}.h5'.format(cfg['data']['type'], cfg['data']['audio_feature'], 
            cfg['data']['sample_rate'], cfg['data']['n_fft'], cfg['data']['hop_length'], cfg['data']['n_mels'])
        scalar_path = scalar_h5_dir.joinpath(fn_scalar)
        with h5py.File(scalar_path, 'r') as hf:
            self.mean = hf['mean'][:]
            self.std = hf['std'][:]
        if cuda:
            self.mean = torch.tensor(self.mean, dtype=torch.float32).cuda()
            self.std = torch.tensor(self.std, dtype=torch.float32).cuda()

        self.label_resolution = dataset.label_resolution
        self.label_interp_ratio = int(self.label_resolution * cfg['data']['sample_rate'] / cfg['data']['hop_length'])
        
    def infer(self, generator):
        fn_list, n_segment_list = [], []
        pred_sed_list, pred_doa_list = [], []

        iterator = tqdm(generator)
        for batch_sample in iterator:
            batch_x = batch_sample['waveform']
            if self.cuda:
                batch_x = batch_x.cuda(non_blocking=True)
            with torch.no_grad():
                self.af_extractor.eval()
                self.model.eval()
                batch_x = self.af_extractor(batch_x)
                batch_x = (batch_x - self.mean) / self.std
                pred = self.model(batch_x)
                pred['sed'] = torch.sigmoid(pred['sed'])
            fn_list.append(batch_sample['filename'])
            n_segment_list.append(batch_sample['n_segment'])
            pred_sed_list.append(pred['sed'].cpu().detach().numpy())
            pred_doa_list.append(pred['doa'].cpu().detach().numpy())

        iterator.close()

        self.fn_list = [fn for row in fn_list for fn in row]
        self.n_segment_list = [n_segment for row in n_segment_list for n_segment in row]
        pred_sed = np.concatenate(pred_sed_list, axis=0)
        pred_doa = np.concatenate(pred_doa_list, axis=0)

        self.num_segments = max(self.n_segment_list) + 1
        origin_num_clips = int(pred_sed.shape[0]/self.num_segments)
        origin_T = int(pred_sed.shape[1]*self.num_segments)
        pred_sed = pred_sed.reshape((origin_num_clips, origin_T, 2, -1))[:, :int(60 / self.label_resolution)]
        pred_doa = pred_doa.reshape((origin_num_clips, origin_T, 2, -1))[:, :int(60 / self.label_resolution)]

        pred = {
            'sed': pred_sed,
            'doa': pred_doa
        }
        return pred

    def fusion(self, submissions_dir, preds):
        """ Ensamble predictions

        """
        num_preds = len(preds)
        pred_sed = []
        pred_doa = []
        for n in range(num_preds):
            pred_sed.append(preds[n]['sed'])
            pred_doa.append(preds[n]['doa'])
        pred_sed = np.array(pred_sed).mean(axis=0)
        pred_doa = np.array(pred_doa).mean(axis=0)

        N, T = pred_sed.shape[:2]
        pred_sed_max = pred_sed.max(axis=-1)
        pred_sed_max_idx = pred_sed.argmax(axis=-1)
        pred_sed = np.zeros_like(pred_sed)
        for b_idx in range(N):
            for t_idx in range(T):
                for track_idx in range(2):
                    pred_sed[b_idx, t_idx, track_idx, pred_sed_max_idx[b_idx, t_idx, track_idx]] = \
                        pred_sed_max[b_idx, t_idx, track_idx]
        pred_sed = (pred_sed > self.cfg['inference']['threshold_sed']).astype(np.float32)
        # convert Catesian to Spherical
        azi = np.arctan2(pred_doa[..., 1], pred_doa[..., 0])
        elev = np.arctan2(pred_doa[..., 2], np.sqrt(pred_doa[..., 0]**2 + pred_doa[..., 1]**2))
        pred_doa = np.stack((azi, elev), axis=-1) # (N, T, tracks, (azi, elev))

        fn_list = self.fn_list[::self.num_segments]
        for n in range(pred_sed.shape[0]):
            fn = fn_list[n]
            pred_sed_f = pred_sed[n][None, ...]
            pred_doa_f = pred_doa[n][None, ...]
            pred_dcase_format_dict = to_dcase_format(pred_sed_f, pred_doa_f)
            csv_path = submissions_dir.joinpath(fn + '.csv')
            self.write_submission(csv_path, pred_dcase_format_dict)
        print('Rsults are saved to {}\n'.format(str(submissions_dir)))

