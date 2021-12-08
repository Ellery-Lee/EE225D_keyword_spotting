import sys
sys.path.append('../')
import pickle
from config.visual_config import getArgs
from dataset_lrw1000 import LRW1000_Dataset as Dataset
import torch
from torch.utils.data import DataLoader
import os
import time
from model import *
from torch.cuda.amp import autocast
from audio_model import AudioModel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

torch.backends.cudnn.benchmark = True

args = getArgs()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

audio_model = AudioModel(args).cuda()


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=False):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        pin_memory=True)
    return loader


def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg


def feature_extractor(split):
    dataset_type = split

    with torch.no_grad():
        dataset = Dataset(dataset_type, args)

        print('Data Length:', len(dataset))
        loader = dataset2dataloader(
            dataset, args.batch_size, args.num_workers, shuffle=False)
        
        # model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h") # superb/wav2vec2-base-superb-ks
        
        print('model prepared, start testing')
        total = 0

        ## open file
        f = open('features/'+dataset_type+'_feature.pkl', 'wb')
        file_appeared = []
        for (i_iter, input) in enumerate(loader):
            filenames = input.get('filename')

            tic = time.time()
            audio = input.get('audio').cuda(non_blocking=True)
            audio_npy = audio.detach().cpu().numpy()
    
            sr = input.get('sr').cuda(non_blocking=True)
            sr_npy = sr.detach().cpu().numpy()
            # label = input.get('label').cuda(non_blocking=True)
            
            total = total + audio.size(0)
            border = input.get('duration').cuda(non_blocking=True).float()
            border_npy = border.detach().cpu().numpy()

            with autocast():
                if(args.border):
                    audioFeature = audio_model(audio_npy, sr_npy, feature_extractor, border_npy)
                else:
                    audioFeature = audio_model(audio_npy, sr_npy, feature_extractor)
                # for-loop store (filename: feature) 
                for i in range(len(filenames)):
                    filename = filenames[i]
                    if filename not in file_appeared:
                        feat = audioFeature[i]
                        pickle.dump((filename, feat), f)
                        file_appeared.append(filename)

            toc = time.time()
            if(i_iter % 10 == 0):
                msg = ''
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)
                              * (len(loader)-i_iter)/3600.0)

                print(msg)
        f.close()


if(__name__ == '__main__'):
    feature_extractor('val_1000_20')
