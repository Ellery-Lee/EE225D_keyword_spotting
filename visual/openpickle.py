import pickle
import os

def processFeatureFiles(category = "trn"):
    filename = category+".txt"
    with open(filename) as info:
        lines = info.readlines()

    fnToWord = {}
    for line in lines:
        lineComponents = line.split(",")
        key = lineComponents[0] # filename
        value = lineComponents[1] # word
        if key not in fnToWord.keys():
            fnToWord[key] = value
        else:
            print("ERROR: duplicate filename:", key)
    f = open('feature.pkl', 'rb')
    # format: ('filename', tensor feature matrix)
    while True:
        try:
            data = pickle.load(f)
            # if in trn.txt
            filename = data[0] # filename
            if filename in fnToWord.keys():
                # get feature and save in the folder with "filename" and the current filename should be a counting number
                word = fnToWord[filename] # new key
                try:
                    if not os.path.exists(word):
                        os.makedirs(word)
                except OSError:
                    print("ERROR: failed to create dir: ", word)
                
                feat = data[1]
                featFile = open(word+'/'+category+'/'+filename+'.pkl', 'wb')
                pickle.dump(feat, featFile)
                featFile.close()
                break

        except EOFError:
            break


