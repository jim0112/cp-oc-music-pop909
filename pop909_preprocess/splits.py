import os, pickle
import numpy as np
midi_dir = 'pop909_melody_midi'
list = os.listdir(midi_dir)
#f_train = open('splits/train.pkl', 'wb')
#f_val = open('splits/train.pkl', 'wb')

t_list = []
v_list = []
for i, l in enumerate(list) :
    if not i % 20 :
        v_list.append(l)
    else :
        t_list.append(l)

#print(np.array(v_list))

t_list = np.array(t_list)
v_list = np.array(v_list)

pickle.dump(
        t_list,
        open('splits/train_pieces.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )

pickle.dump(
        v_list,
        open('splits/val_pieces.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )