import librosa
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        self.args = args

    def forward(self, audio, sr, border=None):
        # scale, sr = librosa.load(audio + '.wav')
        mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram