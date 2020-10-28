import shutil
from functools import reduce
from pathlib import Path
from timeit import default_timer as timer

import h5py
import librosa
import numpy as np
import pandas as pd
import torch
from methods.data import BaseDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.common import float_samples_to_int16
from utils.config import get_afextractor


class Preprocessor:
    """Preprocess the audio data.

    1. Extract wav file and store to hdf5 file
    2. Extract meta file and store to hdf5 file
    """
    def __init__(self, args, cfg, dataset):
        """
        Args:
            args: parsed args
            cfg: configurations
            dataset: dataset class
        """
        self.args = args
        self.cfg = cfg
        self.dataset = dataset

        # Path for dataset
        hdf5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset'])

        # Path for extraction of wav
        self.data_dir_list = [
            dataset.dataset_dir[args.dataset_type]['foa'], 
            dataset.dataset_dir[args.dataset_type]['mic']
        ]
        data_h5_dir = hdf5_dir.joinpath('data').joinpath('{}fs'.format(cfg['data']['sample_rate']))
        self.data_h5_dir_list = [
            data_h5_dir.joinpath(args.dataset_type).joinpath('foa'),
            data_h5_dir.joinpath(args.dataset_type).joinpath('mic')            
        ]
        self.data_statistics_path_list = [
            data_h5_dir.joinpath(args.dataset_type).joinpath('statistics_foa.txt'),
            data_h5_dir.joinpath(args.dataset_type).joinpath('statistics_mic.txt')
        ]

        # Path for extraction of scalar
        self.scalar_h5_dir = hdf5_dir.joinpath('scalar')
        fn_scalar = '{}_{}_sr{}_nfft{}_hop{}_mel{}.h5'.format(cfg['data']['type'], cfg['data']['audio_feature'], 
            cfg['data']['sample_rate'], cfg['data']['n_fft'], cfg['data']['hop_length'], cfg['data']['n_mels'])
        self.scalar_path = self.scalar_h5_dir.joinpath(fn_scalar)

        # Path for extraction of meta
        self.meta_dir = dataset.dataset_dir[args.dataset_type]['meta']
        self.meta_h5_dir = hdf5_dir.joinpath('meta').joinpath(args.dataset_type)

    def extract_data(self):
        """ Extract wave and store to hdf5 file

        """
        print('Converting wav file to hdf5 file starts......\n')
        
        for h5_dir in self.data_h5_dir_list:
            if h5_dir.is_dir():
                flag = input("HDF5 folder {} is already existed, delete it? (y/n)".format(h5_dir)).lower()
                if flag == 'y':
                    shutil.rmtree(h5_dir)
                elif flag == 'n':
                    print("User select not to remove the HDF5 folder {}. The process will quit.\n".format(h5_dir))
                    return
            h5_dir.mkdir(parents=True)
        for statistic_path in self.data_statistics_path_list:
            if statistic_path.is_file():
                statistic_path.unlink()

        for idx, data_dir in enumerate(self.data_dir_list):
            begin_time = timer()
            h5_dir = self.data_h5_dir_list[idx]
            statistic_path = self.data_statistics_path_list[idx]
            audio_count = 0
            silent_audio_count = 0
            data_list = [path for path in sorted(data_dir.glob('*.wav')) if not path.name.startswith('.')]
            iterator = tqdm(data_list, total=len(data_list), unit='it')
            for data_path in iterator:
                # read data
                data, _ = librosa.load(data_path, sr=self.cfg['data']['sample_rate'], mono=False)
                if len(data.shape) == 1:
                    data = data[None,:]
                '''data: (channels, samples)'''

                # silent data statistics
                lst = np.sum(np.abs(data), axis=1) > data.shape[1]*1e-4
                if not reduce(lambda x, y: x*y, lst):
                    with statistic_path.open(mode='a+') as f:
                        print(f"Silent file in feature extractor: {data_path.name}\n", file=f)
                        silent_audio_count += 1
                        tqdm.write("Silent file in feature extractor: {}".format(data_path.name))
                        tqdm.write("Total silent files are: {}\n".format(silent_audio_count))

                # save to h5py
                h5_path = h5_dir.joinpath(data_path.stem + '.h5')
                with h5py.File(h5_path, 'w') as hf:
                    hf.create_dataset(name='waveform', data=float_samples_to_int16(data), dtype=np.int16)

                audio_count += 1

                tqdm.write('{}, {}, {}'.format(audio_count, h5_path, data.shape))

            with statistic_path.open(mode='a+') as f:
                print(f"Total number of audio clips extracted: {audio_count}", file=f)
                print(f"Total number of silent audio clips is: {silent_audio_count}\n", file=f)

            iterator.close()
            print("Extacting feature finished! Time spent: {:.3f} s".format(timer() - begin_time))

        return

    def extract_scalar(self):
        """ Extract scalar and store to hdf5 file

        """
        print('Extracting scalar......\n')
        self.scalar_h5_dir.mkdir(parents=True, exist_ok=True)

        cuda_enabled = not self.args.no_cuda and torch.cuda.is_available()
        train_set = BaseDataset(self.args, self.cfg, self.dataset)
        data_generator = DataLoader(
            dataset=train_set,
            batch_size=32,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        af_extractor = get_afextractor(self.cfg, cuda_enabled).eval()
        iterator = tqdm(enumerate(data_generator), total=len(data_generator), unit='it')
        features = []
        begin_time = timer()
        for it, batch_sample in iterator:
            if it == len(data_generator):
                break
            batch_x = batch_sample['waveform']
            batch_x.require_grad = False
            if cuda_enabled:
                batch_x = batch_x.cuda(non_blocking=True)
            batch_y = af_extractor(batch_x).transpose(0, 1)
            C, _, _, F = batch_y.shape
            features.append(batch_y.reshape(C, -1, F).cpu().numpy())
        iterator.close()
        features = np.concatenate(features, axis=1)
        mean = []
        std = []

        for ch in range(C):
            mean.append(np.mean(features[ch], axis=0, keepdims=True))
            std.append(np.std(features[ch], axis=0, keepdims=True))
        mean = np.stack(mean)[None, ...]
        std = np.stack(std)[None, ...]

        # save to h5py
        with h5py.File(self.scalar_path, 'w') as hf:
            hf.create_dataset(name='mean', data=mean, dtype=np.float32)
            hf.create_dataset(name='std', data=std, dtype=np.float32)
        print("\nScalar saved to {}\n".format(str(self.scalar_path)))
        print("Extacting scalar finished! Time spent: {:.3f} s\n".format(timer() - begin_time))

    def extract_meta(self):
        """ Extract meta .csv file and re-organize the meta data and store it to hdf5 file.

        """
        print('Converting meta file to hdf5 file starts......\n')
        
        shutil.rmtree(str(self.meta_h5_dir), ignore_errors=True)
        self.meta_h5_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg['dataset'] == 'dcase2020task3':
            self.extract_meta_dcase2020task3()
    
    def extract_meta_dcase2020task3(self):
        num_frames = 600
        num_tracks = 2
        num_classes = 14
        meta_list = [path for path in sorted(self.meta_dir.glob('*.csv')) if not path.name.startswith('.')]
        iterator = tqdm(enumerate(meta_list), total=len(meta_list), unit='it')
        for idx, meta_file in iterator:
            fn = meta_file.stem
            df = pd.read_csv(meta_file, header=None)
            sed_label = np.zeros((num_frames, num_tracks, num_classes))
            doa_label = np.zeros((num_frames, num_tracks, 3))
            event_indexes = np.array([[None, None]] * num_frames)  # event indexes of all frames
            track_numbers = np.array([[None, None]] * num_frames)   # track number of all frames
            for row in df.iterrows():
                frame_idx = row[1][0]
                event_idx = row[1][1]
                track_number = row[1][2]                
                azi = row[1][3]
                elev = row[1][4]
                
                ##### track indexing #####
                # default assign current_track_idx to the first available track
                current_event_indexes = event_indexes[frame_idx]
                current_track_indexes = np.where(current_event_indexes == None)[0].tolist()
                current_track_idx = current_track_indexes[0]    

                # tracking from the last frame if the last frame is not empty
                last_event_indexes = np.array([None, None]) if frame_idx == 0 else event_indexes[frame_idx-1]
                last_track_indexes = np.where(last_event_indexes != None)[0].tolist() # event index of the last frame
                last_events_tracks = list(zip(event_indexes[frame_idx-1], track_numbers[frame_idx-1]))
                if last_track_indexes != []:
                    for track_idx in last_track_indexes:
                        if last_events_tracks[track_idx] == (event_idx, track_number):
                            if current_track_idx != track_idx:  # swap tracks if track_idx is not consistant
                                sed_label[frame_idx, [current_track_idx, track_idx]] = \
                                    sed_label[frame_idx, [track_idx, current_track_idx]]
                                doa_label[frame_idx, [current_track_idx, track_idx]] = \
                                    doa_label[frame_idx, [track_idx, current_track_idx]]
                                event_indexes[frame_idx, [current_track_idx, track_idx]] = \
                                    event_indexes[frame_idx, [track_idx, current_track_idx]]
                                track_numbers[frame_idx, [current_track_idx, track_idx]] = \
                                    track_numbers[frame_idx, [track_idx, current_track_idx]]
                                current_track_idx = track_idx
                #########################

                # label encode
                azi_rad, elev_rad = azi * np.pi / 180, elev * np.pi / 180
                sed_label[frame_idx, current_track_idx, event_idx] = 1.0
                doa_label[frame_idx, current_track_idx, :] = np.cos(elev_rad) * np.cos(azi_rad), \
                    np.cos(elev_rad) * np.sin(azi_rad), np.sin(elev_rad)
                event_indexes[frame_idx, current_track_idx] = event_idx
                track_numbers[frame_idx, current_track_idx] = track_number

            meta_h5_path = self.meta_h5_dir.joinpath(fn + '.h5')
            with h5py.File(meta_h5_path, 'w') as hf:
                hf.create_dataset(name='sed_label', data=sed_label, dtype=np.float32)
                hf.create_dataset(name='doa_label', data=doa_label, dtype=np.float32)
            
            tqdm.write('{}, {}'.format(idx, meta_h5_path))         

