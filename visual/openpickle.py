import pickle

with open("trn.txt") as info:
    lines = info.readlines()

names = {}
for line in lines:
    key = line.split(","[1]) # word
    value = []

f = open('feature.pkl', 'rb')
# format: ('filename', tensor feature matrix)

while True:
    try:
        data = pickle.load(f)
        # if in trn.txt
        pdir = data[0]
        if pdir in names:
            # get feature and save in the folder with "filename" and the current filename should be a counting number
            feat = data[1]

    except EOFError:
        break
