# coding: utf-8
dir = '/Users/dongwang/Desktop/225D/project/EE225D_keyword_spotting/config/lrw1000/'
filename = "all_audio_video"
ext = ".txt"
with open(dir+filename+ext, 'r') as f:
    lines = f.readlines()
with open("/Users/dongwang/Desktop/225D/project/EE225D_keyword_spotting/config/lrw1000/first50.txt") as df:
    dflines = df.readlines()

dict = []
for line in dflines:
    word = line.strip('\n')
    dict.append(word)
    
nf = open(filename+"_20"+ext, 'w')

for line in lines:
    if line.split(',')[2] in dict:
        nf.write(line)

nf.close()
