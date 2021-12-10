import pickle
import os

def processFeatureFiles(category = "trn_1000"):
    filename = "../config/lrw1000/visual/" + category + "_20.txt"
    with open(filename) as info:
        lines = info.readlines()
    
    fnToWord = {}
    for line in lines:
        lineComponents = line.split(",")
        key = lineComponents[0] # filename
        value = lineComponents[1] # word
        if key not in fnToWord.keys():
            fnToWord[key] = value

    f = open('features/trn_feature.pkl', 'rb')
    # format: ('filename', tensor feature matrix)
    print("begin creating files-------")

    nfeat = 0
    while True:
        try:
            data = pickle.load(f)
            # if in trn.txt
            filename = data[0] # filename
            filename = filename.split("_")[0]
            if filename in fnToWord.keys():
                nfeat += 1
                print(nfeat)
                # get feature and save in the folder with "filename" and the current filename should be a counting number
                word = fnToWord[filename] # new key
                try:
                    foldername =  '../feature/visual/' + word + '/' + category
                    if not os.path.exists(foldername):
                        os.makedirs(foldername)
                except OSError:
                    print("ERROR: failed to create dir: ", foldername)
                
                feat = data[1]

                for idx in range(999):   
                    featfilename = '../feature/visual/' + word + '/' + category + '/' + filename + "_" + str(idx + 1) +'.pkl'
                    if os.path.exists(foldername + "/" + featfilename):
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
    print("No of features: ", nfeat)

if __name__ == "__main__":
    processFeatureFiles()
    