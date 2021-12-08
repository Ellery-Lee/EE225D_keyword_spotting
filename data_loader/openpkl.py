### this file is only for unit testing

import pickle
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

vpath="/home/dongwang/EE225D_keyword_spotting/features/visual/一/tst_1000/1b9c1fb384ca25fe5c4f4c1e7a5ce44e_1.pkl"
dirlist = ["7fe67cee831a62a64a74de412c8bc7f5", "269c04dac28abe3cfc7f7315df7defaa", "0ab8d4fdd2a9eb55b0b91ec5393e06be", "0e5ee912a00e4f9e979a033fc01b6d21"]
fpath = "/home/dongwang/EE225D_keyword_spotting/feature/audio/一样/tst/7fe67cee831a62a64a74de412c8bc7f5_1.pkl"
test = "/home/dongwang/EE225D_keyword_spotting/feature/audio/扫描/tst_1000/4c48f3c07af503aa7704fbd25fdf0448_1.pkl"

f = open(test, 'rb')
V = pickle.load(f)
print(V)
print(V.shape)
f.close()

# f = open(vpath, 'rb')
# V = pickle.load(f)
# img = V.detach().cpu().numpy()
# img = np.array(img)
# print(img.shape)


# audio, _ = librosa.load("../../LRW1000_Public/audio" + "/" + dirlist[0] + ".wav", sr=16000)
# if len(audio) < 81760:
#     pad = np.zeros(81760 - len(audio))
#     audio = np.concatenate((audio, pad))
# mel_spectrogram = librosa.feature.melspectrogram(audio, sr=16000, n_fft=512, hop_length=160, n_mels=40)
# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
# print(log_mel_spectrogram.shape)

    

# model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h") # superb/wav2vec2-base-superb-ks

# # compute attention masks and normalize the waveform if needed
# inputs = feature_extractor(log_mel_spectrogram, sampling_rate=16000, padding=True, return_tensors="pt")
# print(inputs.get('input_values')[0])
# print(inputs.get('input_values')[0].shape)

# logits = model(**inputs).logits
# print(logits)
# predicted_ids = torch.argmax(logits, dim=-1)
# print(predicted_ids)
# labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
# print(labels)