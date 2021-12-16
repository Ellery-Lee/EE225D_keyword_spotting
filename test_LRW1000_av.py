import matplotlib.pyplot as plt
import argparse
import time
import torch
import random
from tqdm import tqdm
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
import data_loader.dataset_audio_visual as module_data
import model.loss as module_loss
import model.metric as module_met
import model.model_av as module_arch
from torch.utils.data import DataLoader
from utils.util import canonical_state_dict_keys
from parse_config import ConfigParser
from model.metric import AverageMeter
from data_loader.dataset_audio_visual import DatasetV
from torch.utils.data.dataloader import default_collate
import sys
import json
import glob
from scipy.special import expit as sigmoid
from sklearn.metrics import average_precision_score
import pkg_resources
pkg_resources.require("matplotlib==3.4.3")
sys.setrecursionlimit(1500)


def collate_fn(batch):
    if True:
        return batch
    return default_collate(batch)


def calc_eer(scores, labels):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    fnr = fnr*100
    fpr = fpr*100
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    if fpr[idxE] > fnr[idxE]:
        return fpr[idxE]
    else:
        return fnr[idxE]


def mean_average_precision_score(labels, scores, original_labels):
    total_positives = np.sum(original_labels)
    Iscores = np.argsort(-scores)
    labels_sorted = labels[Iscores]
    average_precision_array = []
    counter = 0
    for idx, val in enumerate(labels_sorted):
        if val == 1:
            counter += 1
            average_precision_array.append(counter/float(idx+1))
    mean_average_precision = np.sum(np.asarray(average_precision_array)) / float(total_positives)
    return mean_average_precision


def recall_at_k(r, k, ground_truth):
    assert k >= 1
    r_2 = np.asarray(r)[:k] != 0
    if r_2.size != k:
        raise ValueError('Relevance score length < k')

    return np.sum(r_2)/float(np.sum(ground_truth))


def transform_batch_test(lstV_widx_sent, batchword, config):

    target = []
    lens = []
    vnames = []
    vidx = []
    view = []
    batch_size = len(lstV_widx_sent)
    start_times = []
    end_times = []
    lstV_widx_sent_real = []
    for k in range(0, batch_size):
        if lstV_widx_sent[k][0].size(0) > 1:
            lstV_widx_sent_real.append(lstV_widx_sent[k])
    batch_size = len(lstV_widx_sent_real)
    for k in range(0, batch_size):
        lens.append(lstV_widx_sent_real[k][0].size(0))
        TN = 1 if any(x == batchword for x in lstV_widx_sent_real[k][2]) else 0
        target.append(TN)
        vnames.append(lstV_widx_sent_real[k][3])
        # view.append(lstV_widx_sent_real[k][3])
    lens = np.asarray(lens)
    target = np.asarray(target)
    Ilens = np.argsort(-lens)
    lens = lens[Ilens]
    target = target[Ilens]
    vnames = [vnames[i] for i in Ilens]
    # view = [view[i] for i in Ilens]
    max_len = lens[0]
    max_out_len, rec_field, offset = in2out_idx(max_len)
    batchV = np.zeros(
        (batch_size, max_len, lstV_widx_sent_real[0][0].size(1))).astype('float')
    batchA = np.zeros((batch_size,max_len,lstV_widx_sent_real[0][1].size(1))).astype('float')
    for i in range(0, batch_size):
        batchV[i, :lens[i], :] = lstV_widx_sent_real[Ilens[i]][0].detach().cpu().numpy().copy()
        batchA[i, :lens[i], :] = lstV_widx_sent_real[Ilens[i]][1].detach().cpu().numpy().copy()
    # return batchV, lens, target, vnames, start_times, end_times, rec_field, Ilens
    # batchV.shape (40, 40, 512)
    return batchV, batchA, lens, target, vnames, rec_field, Ilens


def in2out_idx(idx_in):
    layers = [
        {'type': 'conv3d', 'n_channels': 32,  'kernel_size': (1, 5, 5), 'stride': (1, 2, 2), 'padding': (0, 2, 2),
         'maxpool': {'kernel_size': (1, 2, 2), 'stride': (1, 2, 2)}},

        {'type': 'conv3d', 'n_channels': 64, 'kernel_size': (1, 5, 5), 'stride': (1, 2, 2), 'padding': (0, 2, 2),
            # 'maxpool': {'kernel_size' : (1,2,2), 'stride': (1,2,2)}
         },

    ]
    layer_names = None
    from misc.compute_receptive_field import calc_receptive_field
    idx_out, _, rec_field, offset = calc_receptive_field(
        layers, idx_in, layer_names)
    return idx_out, rec_field, offset


def evaluation(config, logger=None):

    if logger is None:
        logger = config.get_logger('test')

    logger.info("Running evaluation with configuration:")
    logger.info(config)
    model = config.init('arch', module_arch)
    logger.info(model)

    tic = time.time()
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = canonical_state_dict_keys(checkpoint['state_dict'])
    model.load_state_dict(state_dict)
    logger.info(f"Finished loading ckpt in {time.time() - tic:.3f}s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    num_words = config["dataset"]["args"]["num_words"]
    num_phoneme_thr = config["dataset"]["args"]["num_phoneme_thr"]
    split = config["dataset"]["args"]["split"]
    cmu_dict_path = config["dataset"]["args"]["cmu_dict_path"]
    data_struct_path_v = config["dataset"]["args"]["data_struct_path_v"]
    data_struct_path_a = config["dataset"]["args"]["data_struct_path_a"]
    p_field_path = config["dataset"]["args"]["field_vocab_paths"]["phonemes"]
    g_field_path = config["dataset"]["args"]["field_vocab_paths"]["graphemes"]
    vis_feat_dir = config["dataset"]["args"]["vis_feat_dir"]
    aud_feat_dir = config["dataset"]["args"]["aud_feat_dir"]
    batch_size = config["data_loader"]["args"]["batch_size"]
    shuffle = config["data_loader"]["args"]["shuffle"]
    drop_last = config["data_loader"]["args"]["drop_last"]
    pin_memory = config["data_loader"]["args"]["pin_memory"]
    num_workers = config["data_loader"]["args"]["num_workers"]
    g2p = config["arch"]["args"]["g2p"]
    use_BE_localiser = config["arch"]["args"]["rnn2"]

    test_dataset = DatasetV(num_words, num_phoneme_thr, cmu_dict_path,
                            vis_feat_dir, aud_feat_dir, split, data_struct_path_v, data_struct_path_a, p_field_path, g_field_path, False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)

    Words = []
    for i, lstVwidx in enumerate(test_loader):
        for b in range(0, len(lstVwidx)):
            for w in lstVwidx[b][2]:
                if w != -1:
                    Words.append(w)

    Words = np.unique(np.asarray(Words).astype('int32')).tolist()
    end = time.time()
    labels = []
    scores = []
    original_labels = []
    names = []

    for j, batchword in enumerate(Words):
        for i, lstVwidx in enumerate(test_loader):
            # inputV, lens, target, vnames, view, start_times, end_times, rec_field, Ilens = transform_batch_test(lstVwidx, batchword, config)
            inputV, inputA, lens, target, vnames, rec_field, Ilens = transform_batch_test(
                lstVwidx, batchword, config)
            names = np.concatenate((names, vnames), axis=0)
            batch_size = inputV.shape[0]   # 40
            widx = np.asarray([batchword]*batch_size).astype('int32')
            target = torch.from_numpy(target).cuda(non_blocking=True)
            inputV = torch.from_numpy(inputV).float().cuda(non_blocking=True)
            inputA = torch.from_numpy(inputA).float().cuda(non_blocking=True)
            widx = torch.from_numpy(widx).cuda(non_blocking=True)
            inputV_var = Variable(inputV)
            inputA_var = Variable(inputA)
            target_var = Variable(target.view(-1, 1)).float()
            grapheme = []
            phoneme = []
            for w in widx:
                grapheme.append(test_dataset.get_GP(w)[0])
                phoneme.append(test_dataset.get_GP(w)[1])
            batchword_str = ''.join(grapheme[0])
            if i == 1:
                logger.info("batchword: {}".format(batchword_str))
            
            graphemeTensor = Variable(
                test_dataset.grapheme2tensor(grapheme)).cuda()
            phonemeTensor = Variable(
                test_dataset.phoneme2tensor(phoneme)).cuda()

            preds = model(vis_feat_lens=lens, p_lengths=None, phonemes=phonemeTensor.detach(),
                            graphemes=graphemeTensor[:-1].detach(), vis_feats=inputV_var, aud_feats=inputA_var, use_BE_localiser=use_BE_localiser, epoch=90, config=config)
            print("finish first stage")

            tdec = graphemeTensor[1:]

            values = preds['max_logit'].view(
                1, len(target)).detach().cpu().numpy()[0]
            ground_truths = target.detach().cpu().numpy()
            original_labels = np.concatenate(
                (original_labels, ground_truths), axis=0)
            if np.where(target.detach().cpu().numpy() == 1)[0].size != 0:
                for k in np.where(target.detach().cpu().numpy() == 1)[0]:
                    logits = []
                    padding = math.ceil((rec_field-1)/2)
                    inputV_loc = inputV[k, :, :].unsqueeze(
                        0).cpu().detach().numpy()
                    inputV_loc = np.pad(inputV_loc, ((
                        0, 0), (padding, padding), (0, 0)), 'constant', constant_values=(0, 0))

                    inputA_loc = inputA[k, :, :].unsqueeze(
                        0).cpu().detach().numpy()
                    inputA_loc = np.pad(inputA_loc, ((
                        0, 0), (padding, padding), (0, 0)), 'constant', constant_values=(0, 0))

                    for m in range(0, lens[k]):
                        inputV_chunck = torch.from_numpy(inputV_loc).float().cuda(
                            non_blocking=True)[:, 11-11+m:11+12+m, :]
                        inputV_var_chunck = Variable(inputV_chunck)
                        inputA_chunck = torch.from_numpy(inputA_loc).float().cuda(
                            non_blocking=True)[:, 11-11+m:11+12+m, :]
                        inputA_var_chunck = Variable(inputA_chunck)
                        lens_loc = np.asarray([23])
                        
                        preds_loc = model(vis_feat_lens=lens_loc, p_lengths=None, phonemes=phonemeTensor[:, k].unsqueeze(1).detach(),
                                            graphemes=graphemeTensor[:-1][:, k].unsqueeze(1).detach(), vis_feats=inputV_var_chunck, aud_feats=inputA_var_chunck,use_BE_localiser=use_BE_localiser, epoch=74, config=config)

                        logits.append(preds_loc["o_logits"][0][1][0].item())
                    logits = np.array(logits)
                    logits = torch.from_numpy(logits)
                    max, index = logits.max(0)
                    # if index < start_times[k]-2 or index > end_times[k]+2:
                    #   assert ground_truths[k] ==1
                    #   ground_truths[k] =0
            labels = np.concatenate((labels, ground_truths), axis=0)
            scores = np.concatenate((scores, values), axis=0)
            print("finish second stage")
    eer = calc_eer(scores, labels)
    recall_k_all_words_1 = []
    recall_k_all_words_5 = []
    recall_k_all_words_10 = []
    map_all_words = []
    print("label length: ", len(labels))
    print("EER {}".format(eer))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None,
                      type=str, help="config file path")
    args.add_argument('--resume', help='path to checkpoint for evaluation',
                      default="data/lili-ckpts/localization_loss_bilstm.pth.tar")
    args.add_argument('--device', type=str, help="indices of GPUs to enable")
    eval_config = ConfigParser(args)
    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, msg
    evaluation(eval_config)