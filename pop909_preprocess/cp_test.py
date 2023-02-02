from beat_align import DEFAULT_RESOLUTION, DEFAULT_TICKS_PER_BEAT
import miditoolkit
from miditoolkit import midi
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from itertools import chain
from chorder import Chord, Dechorder

VELO = np.arange(3, 128, 2)
midi_dir = 'new_pop909_melody_midi'
out_dir = './cp/cp_dataset'

pitches_rec = []

if __name__ == '__main__':
  midis = os.listdir(midi_dir)
  # print (len(midis))
  for mf in midis:
    print ('>> now at', mf)
    midi_obj = miditoolkit.midi.MidiFile(
                  os.path.join(midi_dir, mf)
                )
    chord_event = Dechorder.dechord(midi_obj)
    chord_marker = Dechorder.enchord(midi_obj).markers
    midi_obj.instruments[0].notes = sorted(midi_obj.instruments[0].notes, key=lambda x : x.start)
    n_bars = midi_obj.instruments[0].notes[-1].start // (4 * DEFAULT_TICKS_PER_BEAT)
    n_bars += 1
    print (n_bars)
    bar_pos = []
    notes = []
# 4/4 operation
    flag1 = 0
    flag2 = 0
    pre_bar = -1
    pre_pos = -1
    chord = ''
    index = 0
    for n in midi_obj.instruments[0].notes:
      # track, tempo, chord, bar-beat, type, pitch, duration, velocity
      # type : 0-IGN 1-metrix 2-note
      pitches_rec.append( n.pitch )
      bar = n.start // (4 * DEFAULT_TICKS_PER_BEAT)
      # print (n.start, bar)
      duration = min(n.end - n.start, 4 * DEFAULT_TICKS_PER_BEAT)
      # print (' --', bar, position, duration)
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
      pos = (n.start % (4 * DEFAULT_TICKS_PER_BEAT)) // 120
      Vel_index = np.argmin(np.abs(VELO - n.velocity))
      vel = VELO[Vel_index]
      if bar != pre_bar:
        tmp = [0, 17, 1, 0, 0, 0]
        notes.append(tmp)
        pre_bar = bar
        bar_pos.append(index)
        index += 1
      if pos != pre_pos:
        tmp = [chord, pos + 1, 1, 0, 0, 0]
        notes.append(tmp)
        pre_pos = pos
        index += 1
      tmp = [0, 0, 2, n.pitch - 20, duration / 120, (vel-1) / 2]
      notes.append(tmp)
      index += 1
    pickle.dump(
       (bar_pos, notes),
       open(os.path.join(out_dir, mf.split('_')[0] + '.pkl'), 'wb'),
       protocol=pickle.HIGHEST_PROTOCOL
    )
    print("done!")
  print("all finished!")
  #making cp_Chord vocab
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
    open('pop909_cp_Chord_vocab.pkl', 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL
  )
