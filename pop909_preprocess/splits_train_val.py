from shutil import copyfile
import os, pickle

trg_dir = 'splits'
src_dir = 'pop909_melody_midi'
train = 'train'
val = 'val'
if not os.path.exists(os.path.join(trg_dir, train + '/')) :
    os.makedirs(os.path.join(trg_dir, train + '/'))
if not os.path.exists(os.path.join(trg_dir, val + '/')) :
    os.makedirs(os.path.join(trg_dir, val + '/'))

cur = os.listdir(src_dir)

for i,c in enumerate(cur) :
    if not i % 20 :
        copyfile(os.path.join(src_dir, c), os.path.join(trg_dir, val, c))
    else :
        copyfile(os.path.join(src_dir, c), os.path.join(trg_dir, train, c))