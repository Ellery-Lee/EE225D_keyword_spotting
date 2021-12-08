import pandas as pd
import numpy as np
import json
import pinyin
file = open("all_audio_video.txt", "r")
lines = file.readlines()

graphemes = []
phonemes = []

for line in lines:
  data = line.split(",")
  graphemes.append(data[2])
  phonemes.append(data[3])

dict_w_freq = {}
for i in range(len(graphemes)):
  if graphemes[i] not in dict_w_freq.keys():
    dict_w_freq[graphemes[i]] = 0
  dict_w_freq[graphemes[i]] = dict_w_freq[graphemes[i]] + 1

with open("w_freq.json", "w") as outfile:
    json.dump(dict_w_freq, outfile)

# dictionary (word, pinyin)
dict_w_py = {}
for word in dict_w_freq.keys():
  py = pinyin.get(word, format="numerical", delimiter=" ")
  dict_w_py[word] = py

with open("w_py.json", "w") as outfile:
    json.dump(dict_w_py, outfile)

# grapheme (single characters)
dict_g_freq = {}
for word in dict_w_freq.keys():
  w_freq = dict_w_freq[word]
  for i in range(len(word)):
    if word[i] not in dict_g_freq.keys():
      dict_g_freq[word[i]] = 0
    dict_g_freq[word[i]] = dict_g_freq[word[i]] + w_freq

with open("g_freq.json", "w") as outfile:
    json.dump(dict_g_freq, outfile)

g_dict = ["<unk>", "<pad>", "<s>"]
g_dict = g_dict + (list(dict_g_freq.keys()))
g_dict.append("</s>")

with open("graphemes.json", "w") as outfile:
    json.dump(g_dict, outfile)

# phoneme (pronounciation of a single characters)
dict_py_freq = {}
for word, py in dict_w_py.items():
  curr_py = py.split(" ")
  w_freq = dict_w_freq[word]
  for i in range(len(curr_py)):
    if curr_py[i] not in dict_py_freq.keys():
      dict_py_freq[curr_py[i]] = 0
    dict_py_freq[curr_py[i]] = dict_py_freq[curr_py[i]] + w_freq

with open("py_freq.json", "w") as outfile:
    json.dump(dict_py_freq, outfile)

p_dict = ["<unk>", "<pad>", "<os>", "</os>"]
p_dict = p_dict + (list(dict_py_freq.keys()))

with open("phonemes.json", "w") as outfile:
    json.dump(p_dict, outfile)

### statistics

print("Total unique words: " + str(len(dict_w_freq)))
print("Total unique graphemes: " + str(len(dict_g_freq)))
print("Total unique phonemes: " + str(len(dict_py_freq)))

f = open('w_py.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
words = list(data.keys())

dict = {}
for idx, word in enumerate(words):
  dict[word] = idx

with open("tst_1000.txt") as f:
  lines = f.readlines()

idx_arr = []
for line in lines:
  elems = line.split(",")
  word = elems[1]
  idx = dict[word]
  if idx not in idx_arr:
    idx_arr.append(idx)

val_dict = {}
val_dict['widx'] = idx_arr

with open("LRW1000_test_words.json", "w") as outfile:
    json.dump(val_dict, outfile)

len(val_dict['widx'])

textfile = open("LRW1000words.txt", "w")
for element in words:
    textfile.write(element + "\n")
textfile.close()

f = open('w_py.json')
data = json.load(f)

textfile = open("LRWDict.txt", "w")
count = 0
for key, value in data.items():
  count += 1
  textfile.write(key + " " + value + "\n")
  print(str(count) + "key is" + str(key) + ", txt is " + str(key) + " " + str(value))
textfile.close
print(count)

with open('LRWDict.txt') as f:
  lines = f.readlines()