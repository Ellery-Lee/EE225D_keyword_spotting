import sys
sys.path.append('../')
import pickle
from config.visual_config import getArgs
from visual.dataset_lrw1000 import LRW1000_Dataset as Dataset
from visual.video_model import VideoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import time
from model import *
import torch.optim as optim
from torch.cuda.amp import autocast


torch.backends.cudnn.benchmark = True

args = getArgs()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

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

        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(
            dataset, args.batch_size, args.num_workers, shuffle=False)


        ## open file
        f = open('features/trn_feature.pkl', 'wb')

        for (i_iter, input) in enumerate(loader):
            filenames = input.get('filename')

            video_model.eval()

            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            
            total = total + video.size(0)
            border = input.get('duration').cuda(non_blocking=True).float()

            with autocast():
                if(args.border):
                    f_v, y_v = video_model(video, border)
                else:
                    f_v, y_v = video_model(video)
                # for-loop store (filename: feature) 
                for i in range(len(filenames)):
                    filename = filenames[i]
                    if filename not in file_appeared:
                        feat = f_v[i]
                        pickle.dump((filename, feat), f)
                        file_appeared.append(filename)

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if(i_iter % 10 == 0):
                msg = ''
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(
                    v_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)
                              * (len(loader)-i_iter)/3600.0)

                print(msg)
        f.close()

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)

        return acc, msg


if(__name__ == '__main__'):
    acc, msg= feature_extractor('test')
    print(f'acc={acc}')
    exit()
