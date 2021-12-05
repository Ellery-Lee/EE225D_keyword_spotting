dir = '../config/lrw1000/'
visualDir = 'visual/'
audioDir = 'audio/'
trainFileName = 'trn_1000.txt'
validFileName = 'val_1000.txt'
testFileName = 'tst_1000.txt'
mapFile = 'all_audio_video.txt'

visualToAudio = {}
with open(dir+mapFile) as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(',')
        key = line[0]+line[4].rstrip('0')+line[5].rstrip('\n').rstrip('0')
        value = line[1]
        visualToAudio[key] = value
f.close()
validfile = open(dir+audioDir+validFileName, 'w')
with open(dir+visualDir+validFileName) as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(',')
        key = line[0]+line[3].rstrip('0')+line[4].rstrip('\n').rstrip('0')
        newLine = [visualToAudio[key]]+line[1:]
        newLine = ','.join(str(i) for i in newLine)
        validfile.write(newLine)
f.close()
validfile.close()
