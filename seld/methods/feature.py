import torch
import torch.nn as nn

from methods.utils.stft import (STFT, LogmelFilterBank, intensityvector,
                                spectrogram_STFTInput)


class LogmelIntensity_Extractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        data = cfg['data']
        sample_rate, n_fft, hop_length, window, n_mels, fmin, fmax = \
            data['sample_rate'], data['n_fft'], data['hop_length'], data['window'], data['n_mels'], \
                data['fmin'], data['fmax']
        
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # STFT extractor
        self.stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, 
            window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=data['feature_freeze'])
        
        # Spectrogram extractor
        self.spectrogram_extractor = spectrogram_STFTInput
        
        # Logmel extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=data['feature_freeze'])

        # Intensity vector extractor
        self.intensityVector_extractor = intensityvector

    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins)
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x = self.stft_extractor(x)
        logmel = self.logmel_extractor(self.spectrogram_extractor(x))
        intensity_vector = self.intensityVector_extractor(x, self.logmel_extractor.melW)
        out = torch.cat((logmel, intensity_vector), dim=1)
        return out

