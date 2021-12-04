### this file is only for unit testing

import pickle
## feature/visual/条件/trn_1000/8710dbe246c62991a83db83657d4c969_1.pkl
fpath = "../feature/visual/条件/trn_1000/8710dbe246c62991a83db83657d4c969_1.pkl"
f = open(fpath, 'rb')
V = pickle.load(f)
print(V)
