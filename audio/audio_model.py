import librosa
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        self.args = args

    def forward(self, batch_audio, batch_sr, feature_extractor, batch_border=None):
        # scale, sr = librosa.load(audio + '.wav')
        feat = []
        for i in range(len(batch_audio)):
            audio = batch_audio[i]
            sr = batch_sr[i]
            # mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=160, n_mels=10, window="hamming", win_length=512)
            mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=160, n_mels=40)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

            inputs = feature_extractor(log_mel_spectrogram, sampling_rate=16000, padding=True, return_tensors="pt")
            inputs = inputs.get("input_values")[0].detach().cpu().numpy()
            feat.append(inputs)
        return torch.Tensor(np.array(feat))