import os, pickle, random, copy
import numpy as np
import miditoolkit
dir = 'octuple/octuple_dataset/001.pkl'
piece = pickle.load(open(dir, 'rb'))[1]
print(piece)
cp_dir = 'cp/cp_dataset/001.pkl'
cp_piece = pickle.load(open(cp_dir, 'rb'))[1]
print(cp_piece)
    
