import librosa
from torch.utils.data import Dataset
# import cv2
import os
import numpy as np
import torch
# from collections import defaultdict
from torch.utils.data import DataLoader
# from turbojpeg import TurboJPEG

# jpeg = TurboJPEG()


class LRW1000_Dataset(Dataset):

    def __init__(self, split, target_dir):
        self.data = []
        self.index_file = '../config/lrw1000/audio/' + split + '.txt'
        self.target_dir = target_dir
        lines = []

        with open(self.index_file, 'r') as f:
            lines.extend([line.strip().split(',') for line in f.readlines()])

        self.padding = 0
        pinyins = sorted(np.unique([line[2] for line in lines]))
        self.data = [(line[0], int(float(line[3])*25)+1, int(float(line[4])
                      * 25)+1, pinyins.index(line[2])) for line in lines] # audioFileName,op,ed,pinyinIndex

        self.lengths = [data[2]-data[1] for data in self.data]
        self.pinyins = pinyins

    def __len__(self):
        return len(self.data)

    def load_audio(self, item):
        # load audio into a tensor
        (path, op, ed, label) = item

        try:
            inputs, sr = librosa.load('../../LRW1000_Public/audio/' + path + '.wav', sr=16000)
        except:
            inputs = np.zeros()
            sr = 16000
        maxlength = 42400
        if len(inputs) < maxlength:
            pad = np.zeros(maxlength - len(inputs))
            inputs = np.concatenate((inputs, pad))
        border = self.getBorder(op, ed)
        result = {}
        result['filename'] = path
        result['audio'] = inputs
        result['label'] = int(label)
        result['duration'] = border.astype(np.bool)
        result['sr'] = sr

        return result

    def __getitem__(self, idx):
        r = self.load_audio(self.data[idx])

        return r

    def getBorder(self, op, ed):
        center = (op + ed) / 2
        length = (ed - op + 1)
        op = int(center - self.padding // 2)
        ed = int(op + self.padding)
        left_border = max(int(center - length / 2 - op), 0)
        right_border = min(int(center + length / 2 - op), self.padding)
        border = np.zeros((40))
        border[left_border:right_border] = 1.0
        return border

        center = (op + ed) / 2
        length = (ed - op + 1)

        op = int(center - self.padding // 2)
        ed = int(op + self.padding)
        left_border = max(int(center - length / 2 - op), 0)
        right_border = min(int(center + length / 2 - op), self.padding)

        files = [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
        files = filter(lambda path: os.path.exists(path), files)
        files = [cv2.imread(file) for file in files]
        files = [cv2.resize(file, (96, 96)) for file in files]

        files = np.stack(files, 0)
        t = files.shape[0]

        tensor = np.zeros((40, 96, 96, 3)).astype(files.dtype)
        border = np.zeros((40))
        tensor[:t, ...] = files.copy()
        border[left_border:right_border] = 1.0

        tensor = [jpeg.encode(tensor[_]) for _ in range(40)]

        return tensor, border


if(__name__ == '__main__'):
    import soundfile as sf
    
    dir = '../../LRW1000_Public/audio'

    max_audio_length = 0
    dirlist = os.listdir(dir)
    for i in range(len(dirlist)):
        audio, sr = librosa.load(dir + "/" + dirlist[i], sr=16000)
        max_audio_length = max(max_audio_length, len(audio))
        if i % 100 == 0:
            print("processing", i)
            print("maxlength: ", max_audio_length)
    print(max_audio_length)
