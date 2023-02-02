import miditoolkit
from miditoolkit import midi
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from itertools import chain
from chorder import Chord, Dechorder
src_dir = './new_pop909_melody_midi'
out_dir = './octuple/octuple_dataset'

VELO = np.arange(3, 128, 2)
midis = os.listdir(src_dir)
max_bars = -1
for piece in midis:
  print ('>> now at', piece)
  midi_obj = miditoolkit.midi.MidiFile(os.path.join(src_dir, piece))
  chord_event = Dechorder.dechord(midi_obj)
  chord_marker = Dechorder.enchord(midi_obj).markers
  midi_obj.instruments[0].notes = sorted(midi_obj.instruments[0].notes, key=lambda x : x.start)
  n_bars = midi_obj.instruments[0].notes[-1].start // (4 * 480)
  n_bars += 1
  print (n_bars)
  x = []
  bar_st_rec = []
  pre_bar = -1
  flag1 = 0
  flag2 = 0
  chord = '0'
  for index, n in enumerate(midi_obj.instruments[0].notes):
    tmp = []
    tmp.append(n.pitch - 20)
    Vel_index = np.argmin(np.abs(VELO - n.velocity))
    vel = VELO[Vel_index]
    tmp.append(int((vel - 1) / 2))
    duration = min(n.end - n.start, 4 * 480)
    tmp.append(int(duration / 120))
    if flag1 < len(chord_marker) and chord_marker[flag1].time <= n.start:
      #print(chord_marker[flag1].time, n.start)
      if chord_event[flag2].quality == None and chord_event[flag2].root_pc == None:
        chord = 'N_N'
      else:
        chord = '{}_{}'.format(chord_event[flag2].root_pc, chord_event[flag2].quality)
      while flag2 < len(chord_event)-1 and chord_event[flag2] == chord_event[flag2+1]:
        flag2 += 1
      flag2 += 1
      flag1 += 1
    tmp.append(chord)
    position = ( n.start % (4 * 480) ) // 120
    tmp.append(int(position) + 1)
    bar = n.start // (4 * 480)
    tmp.append(int(bar) + 1)
    if bar != pre_bar:
      bar_st_rec.append(index)
      pre_bar = bar
    x.append(tmp)
    if pre_bar > max_bars:
      max_bars = pre_bar
  print(bar_st_rec)
  pickle.dump(
      (bar_st_rec, x),
      open(os.path.join(out_dir, piece.split('_')[0] + '.pkl'), 'wb'),
      protocol=pickle.HIGHEST_PROTOCOL
  )
  print('done!!')

print(max_bars)
print('start making vocab!!==================')
chord_vocab = []
chord_vocab.append('IGN')
chord_vocab.append('N_N')
for chord in Chord.standard_qualities:
  for i in range(0, 12):
    chord_vocab.append('{}_{}'.format(i, chord))
event2idx, idx2event = dict(), dict()
for i, ev in enumerate(chord_vocab):
  event2idx[ ev ] = i
  idx2event[ i ] = ev
  print('{:03d} : {}'.format(i, idx2event[i]))
pickle.dump(
  tuple((event2idx, idx2event)),
  open('pop909_octuple_Chord_vocab.pkl', 'wb'),
  protocol=pickle.HIGHEST_PROTOCOL
)


