import librosa
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np


class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        self.args = args

    def forward(self, batch_audio, batch_sr, batch_border=None):
        # scale, sr = librosa.load(audio + '.wav')
        feat = []
        for i in range(len(batch_audio)):
            audio = batch_audio[i]
            sr = batch_sr[i]
            mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            feat.append(log_mel_spectrogram)
        return torch.Tensor(np.array(feat))