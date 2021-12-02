import argparse
import os
import h5py
import numpy as np
import torch
import torchtext.data as data
import torch.utils.data
import sys
from tqdm import tqdm
import json

def get_CMU_words(CMU_path):
  words = []
  with open(CMU_path) as f:
    lines = f.readlines()
  for _, line in enumerate(lines):
    line = line.strip("\n")
    words.append(line)
  return words

def get_LRW_split(args, split, CMUwords):
  lst_path = args.LRW_words_path
  word_indices = []
  with open(lst_path) as f:
    lines = f.readlines() #list of words
  for word in tqdm(lines):
    word = word.strip("\n")
    word = word.strip()
    widx_array = []
    widx = CMUwords.index(word)    
    widx_array.append(widx)
    for filename in os.listdir(os.path.join(args.LRW_path, word, split)):
      Fwidx = {}
      L = filename.strip()
      path = os.path.join(word, split, L).replace(".pkl", "")
      Fwidx['widx']=widx_array
      Fwidx['fn']=path
      word_indices.append(Fwidx)
      return word_indices
  return word_indices

def get_LRW_splits():
  parser = argparse.ArgumentParser(description='Script for creating main splits of LRW.')
  parser.add_argument('--CMUdict_path', default='../data/LRW1000words.txt')
  parser.add_argument('--LRW_path', default='../data/lrw_1000/features/main/') 
  parser.add_argument('--LRW_words_path', default='../data/LRW1000words.txt')
  args = parser.parse_args()
  CMUwords = get_CMU_words(args.CMUdict_path)    
  S = ['trn_1000', 'val_1000', 'tst_1000']
  Dsplits = {}
  for i,s in enumerate(S):
    Dsplits[s] = get_LRW_split(args, s, CMUwords)
    break
  with open("../data/lrw_1000/DsplitsLRW1000.json", "w") as fp:
    json.dump(Dsplits, fp)

if __name__=='__main__':
  get_LRW_splits()