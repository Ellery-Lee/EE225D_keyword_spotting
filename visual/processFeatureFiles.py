import pickle
import os

def processFeatureFiles(category = "trn_1000"):
    filename = "../../LRW1000_Public/info/" + category + ".txt"
    with open(filename) as info:
        lines = info.readlines()

    fnToWord = {}
    for line in lines:
        lineComponents = line.split(",")
        key = lineComponents[0] # filename
        value = lineComponents[1] # word
        if key not in fnToWord.keys():
            fnToWord[key] = value

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
                    foldername =  word + '/' + category
                    if not os.path.exists(foldername):
                        os.makedirs(foldername)
                except OSError:
                    print("ERROR: failed to create dir: ", foldername)
                
                feat = data[1]

                for idx in range(999):   
                    featfilename = word + '/' + category + '/' + filename + "_" + str(idx + 1) +'.pkl'
                    if os.path.exists(featfilename):
                        continue
                    else:
                        featFile = open(featfilename, 'wb')
                        pickle.dump(feat, featFile)
                        featFile.close()
                        break
                else:
                    print("ERROR: more than 1000 samples")
                    exit()
        except EOFError:
            break

if __name__ == "__main__":
    processFeatureFiles()