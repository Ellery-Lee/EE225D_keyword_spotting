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

video_model = VideoModel(args).cuda()
print('-------')

def parallel_model(model):
    model = nn.DataParallel(model)
    return model


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items(
    ) if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items(
    ) if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optim_video = optim.Adam(video_model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optim_video, T_max=args.max_epoch, eta_min=5e-6)


if(args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))
    load_missing(video_model, weight.get('video_model'))


video_model = parallel_model(video_model)


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
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


def feature_extractor(is_train):
    dataset_type = 'train' if is_train else 'val'

    with torch.no_grad():
        dataset = Dataset(dataset_type, args)

        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(
            dataset, args.batch_size, args.num_workers, shuffle=False)

        print('start testing')
        v_acc = []
        total = 0

        ## open file
        f = open('feature.pkl', 'wb')

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
                print(f_v.shape)
                # for-loop store (filename: feature) 
                for i in range(len(filenames)):
                    filename = filenames[i]
                    feat = f_v[i]
                    pickle.dump((filename, feat), f)

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
    acc, msg= feature_extractor(True)
    print(f'acc={acc}')
    exit()
