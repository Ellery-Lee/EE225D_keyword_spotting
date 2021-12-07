dir = '/Users/dongwang/Desktop/225D/project/EE225D_keyword_spotting/config/lrw1000/'
filename = "all_audio_video"
ext = ".txt"
with open(dir+filename+ext, 'r') as f:
    lines = f.readlines()

nf = open(filename+"_20"+ext, 'w')

for line in lines:
    if line.split(',')[0] == '19e5c24d226c1f8f4bd2490db172649e' and line.split(',')[1] == 'd31e32048b51f66775ae2be3636a52ac':
        break
    nf.write(line)
nf.close()
