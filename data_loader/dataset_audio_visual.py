import os
import json
import argparse
import tqdm
import torch
import torchtext
import pickle
import numpy as np
import h5py
from pathlib import Path
import torch.utils.data
import torchtext.legacy.data as data
import re
from torchvision import datasets, transforms
from base import BaseDataLoader
from base import BaseDataset
from torch.utils.data.dataloader import default_collate


class CMUDict(torchtext.legacy.data.Dataset):
    def __init__(self, data_lines, word_indices, i_field, g_field, p_field):
        fields = [('idx', i_field), ('grapheme', g_field), ('phoneme', p_field)]
        examples = []  
        wcnt = 0        
        for wcnt, line in enumerate(data_lines):
            grapheme, phoneme = line.split(" ",1)
            examples.append(data.Example.fromlist([word_indices[wcnt], grapheme, phoneme], fields))
        super().__init__(examples, fields)

    @classmethod
    def splits_datasetv_test(cls, cmu_dict_path, i_field, g_field, p_field):
      with open(cmu_dict_path) as f:
        lines = f.readlines()

      with open('/home/dongwang/EE225D_keyword_spotting/data/20/lrw_1000/LRW1000_test_words.json', "r") as fp:
        widx_object = json.load(fp)
        test_words = widx_object['widx']
      test_lines = []
      for i in test_words:
        test_lines.append(lines[i])
      test_data = cls(
         data_lines = test_lines,
         word_indices = test_words,
         i_field = i_field,
         g_field = g_field,
         p_field = p_field
      )
      return test_data
    
    @classmethod
    def splits_datasetv_val(cls, cmu_dict_path, i_field, g_field, p_field):
      with open(cmu_dict_path) as f:
        lines = f.readlines()

      with open('/home/dongwang/EE225D_keyword_spotting/data/20/lrw_1000/LRW1000_val_words.json', "r") as fp:
        widx_object = json.load(fp)
        val_words = widx_object['widx']
      val_lines = []
      for i in val_words:
        val_lines.append(lines[i])
      val_data = cls(
         data_lines = val_lines,
         word_indices = val_words,
         i_field = i_field,
         g_field = g_field,
         p_field = p_field
      )
      return val_data
    
    
    @classmethod
    def splits_dataset_train(cls, cmu_dict_path, i_field, g_field, p_field): 
      with open(cmu_dict_path) as f:
        lines = f.readlines()

      with open('/home/dongwang/EE225D_keyword_spotting/data/20/lrw_1000/LRW1000_train_words.json', "r") as fp:
        widx_object = json.load(fp)
        widx = widx_object['widx'] 
      data_lines = []
      for i in widx:
        data_lines.append(lines[i])
      data = cls(
          data_lines = data_lines,
          word_indices = widx,
          i_field = i_field,
          g_field = g_field,
          p_field = p_field,
      )
      return data
  
def get_splits_datasetv(cmu_dict_path, data_struct_path, splitname):
    i_field = data.Field(lambda x: x)
    g_field = data.Field(init_token='<s>',
                     tokenize=(lambda x: list(x.split('(')[0]))) #sequence reversing removed                                                                   
    p_field = data.Field(init_token='<os>', eos_token='</os>',
                     tokenize=(lambda x: x.split('#')[0].split()))

    if splitname == "trn_1000":
      Wstruct = CMUDict.splits_dataset_train(cmu_dict_path, i_field, g_field, p_field)    
    elif splitname == "tst_1000":                  
      Wstruct = CMUDict.splits_datasetv_test(cmu_dict_path, i_field, g_field, p_field)
    else:
      Wstruct = CMUDict.splits_datasetv_val(cmu_dict_path, i_field, g_field, p_field)
    return Wstruct

def merge_train_pretrain(Dstruct):
    for s in Dstruct['pretrain']:
      Dstruct['train'].append(s)
    return Dstruct
 
class DatasetV(BaseDataset):
 
    def __init__(self, num_words, num_phoneme_thr, cmu_dict_path, Vpath, Apath,
        splitname, data_struct_path_v, data_struct_path_a, p_field_vocab_path, g_field_vocab_path,merge):

        self.data_struct_path_v = data_struct_path_v
        self.data_struct_path_a = data_struct_path_a

        with open(data_struct_path_v, 'r') as f:
          Dstructv = json.load(f)
        with open(data_struct_path_a, 'r') as f:
          Dstructa = json.load(f)
        if "trn_1000" in Dstructv.keys(): 
          self.Ntrain = len(Dstructv["trn_1000"]) 
        if merge == True:         
          Dstructv = merge_train_pretrain(Dstructv) 

        self.Dstructv = Dstructv[splitname]
        self.Dstructa = Dstructa[splitname]
        self.splitname = splitname
        self.Vpath = Vpath 
        self.Apath = Apath 
        self.wstruct = get_splits_datasetv(cmu_dict_path,data_struct_path_v, splitname) # data_struct_path_v doesn't change anything
        self.num_phoneme_thr = num_phoneme_thr
        self.num_words = num_words
        super().__init__(self.num_words, self.wstruct, self.num_phoneme_thr)
        self.word_mask, self.word_indices = self.set_word_mask()
        self.length = len(self.Dstructv)

        with open(g_field_vocab_path, 'r') as f:
            self.g_field_vocab = json.load(f)
        with open(p_field_vocab_path, 'r') as f:
            self.p_field_vocab = json.load(f)
        msg = "g_size: {}, p_size: {}"
        self.g_size = len(self.g_field_vocab)
        self.p_size = len(self.p_field_vocab)
        print(msg.format(self.g_size, self.p_size))  

    def __getitem__(self, index):
        Didx = 0
        if self.splitname == 'trn_1000' and index>=self.Ntrain: 
          Didx = 1  
        if Didx < len(self.Vpath) and index < len(self.Dstructv):
          vfpath = os.path.join(self.Vpath[Didx],self.Dstructv[index]['fn']+'.pkl')
          afpath = os.path.join(self.Apath[Didx],self.Dstructa[index]['fn']+'.pkl')
        else:
          return self.__getitem__(index+1)

        if not os.path.isfile(vfpath) or not os.path.isfile(afpath):
          return self.__getitem__(index+1)
        # read from pkl file
        fv = open(vfpath, 'rb')
        V = pickle.load(fv)

        fa = open(afpath, 'rb')
        A = pickle.load(fa)

        # V = np.load(fpath)
        # V = torch.from_numpy(V).float()

        if V.size()[0]>500:
          return self.__getitem__(index+1)
        widx = self.Dstructv[index]['widx']
        # if 'start_word' in self.Dstruct[index].keys():
        #   start_times = self.Dstruct[index]['start_word'] 
        #   end_times = self.Dstruct[index]['end_word']
        for w in range(0,len(widx)):
            if widx[w] == -1 or self.word_mask[widx[w]]==False:
                widx[w] = -1
        # if 'start_word' in self.Dstruct[index].keys(): 
        #     self.Dstruct[index]['view'] = 'UK'
        #     return V, widx, self.Dstruct[index]['fn'], self.Dstruct[index]['view'], start_times, end_times
        # else:
        #     return V, widx, self.Dstruct[index]['fn']
        return V, A, widx, self.Dstructv[index]['fn']
        
    def grapheme2tensor(self, grapheme):
        mlen = 0
        for i, w in enumerate(grapheme):
            if mlen < len(w):
                mlen = len(w)
        G = np.ones((mlen + 2, len(grapheme)), dtype='int64')
        G[:, :] = self.g_field_vocab.index('<pad>')
        gs = self.g_field_vocab.index('<s>')
        ge = self.g_field_vocab.index("</s>")
        for i, w in enumerate(grapheme):
            G[0, i] = gs
            for j, g in enumerate(w):
                G[j + 1, i] = self.g_field_vocab.index(g)
                G[j + 2, i] = ge
        return torch.from_numpy(G)

    def grapheme2tensor_g2p(self, grapheme):
        mlen = 0
        for i,w in enumerate(grapheme):
            if mlen<len(w):
                mlen = len(w)
        G = np.ones((mlen+1,len(grapheme)),dtype='int64')
        G[:,:] = self.g_field_vocab.index('<pad>')
        gs = self.g_field_vocab.index('<s>')
        for i,w in enumerate(grapheme):
            for j,g in enumerate(w):
                G[j,i] = self.g_field_vocab.index(g)
                G[j+1,i] = gs
        G = np.flip(G,0).copy()
        return torch.from_numpy(G)

    def phoneme2tensor(self, phoneme):
        mlen = 0
        for i, w in enumerate(phoneme):
            if mlen < len(w):
                mlen = len(w)
        P = np.ones((mlen + 1, len(phoneme)), dtype='int64')
        P[:, :] = self.p_field_vocab.index('<pad>')
        ps = self.p_field_vocab.index('<os>')
        for i, w in enumerate(phoneme):
            for j, p in enumerate(w):
                P[j, i] = self.p_field_vocab.index(p)
                P[j + 1, i] = ps
        P = np.flip(P, 0).copy()
        return torch.from_numpy(P)

    def phoneme2tensor_g2p(self, phoneme):
        mlen = 0
        for i,w in enumerate(phoneme):
            if mlen<len(w):
                mlen = len(w)
        P = np.ones((mlen+2,len(phoneme)),dtype='int64')
        P[:,:] = self.p_field_vocab.index('<pad>')
        ps = self.p_field_vocab.index('<os>')
        pe = self.p_field_vocab.index('</os>')
        for i,w in enumerate(phoneme):
            P[0,i] = ps
            for j,p in enumerate(w):
                P[j+1,i] = self.p_field_vocab.index(p)
                P[j+2,i] = pe
        return torch.from_numpy(P)

if  __name__ == "__main__":
    main()