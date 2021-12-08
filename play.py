import json

filename = "data/100/lrw_1000/DsplitsLRW1000.json"

with open(filename, 'r') as f:
    Dstruct = json.load(f)
tst = Dstruct['tst_1000']
print(len(tst))