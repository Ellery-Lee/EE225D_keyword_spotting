import json

with open('/home/dongwang/EE225D_keyword_spotting/config/lrw1000/all_audio_video_20.txt', 'r') as f:
  lines = f.readlines()

audioTovideo = {}
for line in lines:
  elem = line.split(",")
  v = elem[0]
  a = elem[1]
  audioTovideo[a] = v

fa = open("DsplitsLRW1000_audio.json")
audio = json.load(fa)

fv = open("DsplitsLRW1000_visual_deprecated.json")
visual = json.load(fv)

splits = ['trn_1000', 'val_1000', 'tst_1000']

aligned_vinfo = {}
aligned_ainfo = {}
for split in splits:
    alist = audio[split]
    vlist = visual[split]
    vfnlist = {}
    aligned_vlist = []
    aligned_alist = []
    count = 0
    for velem in vlist:
        widx = velem['widx']
        fvpath = velem['fn'].split('/')[2]
        fvpath = fvpath.split('_')[0]           # visual file name
        key = str(widx) + '_' + fvpath
        vfnlist[key] = velem
    
    for aelem in alist:
        widx = aelem['widx']
        fapath = aelem['fn'].split('/')[2]
        fapath = fapath.split('_')[0]           # audio file name
        fvpath = audioTovideo[fapath]
        key = str(widx) + '_' + fvpath
        if key in vfnlist.keys():
            aligned_vlist.append(vfnlist[key])
            aligned_alist.append(aelem)
            count = count + 1
    print(count)
    aligned_vinfo[split] = aligned_vlist
    aligned_ainfo[split] = aligned_alist

out_file = open("test_visual.json", "w")
  
json.dump(aligned_vinfo, out_file)
  
out_file.close()

out_file = open("test_audio.json", "w")
  
json.dump(aligned_ainfo, out_file)
  
out_file.close()

