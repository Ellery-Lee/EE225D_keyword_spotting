import pandas as pd
import numpy as np
import json
# import pinyin
file = open("all_audio_video.txt", "r")
lines = file.readlines()

graphemes = []
phonemes = []

for line in lines:
  data = line.split(",")
  graphemes.append(data[2])
  phonemes.append(data[3])