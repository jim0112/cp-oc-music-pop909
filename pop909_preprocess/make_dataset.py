from beat_align import DEFAULT_RESOLUTION, DEFAULT_TICKS_PER_BEAT
import miditoolkit
from miditoolkit import midi
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from itertools import chain
from chorder import Chord, Dechorder

TEMPOS = np.arange(32, 227, 3)
midi_dir = 'new_pop909_melody_midi'
out_dir = 'new_pop909_with_bars'

pitches_rec = []
n_bars_rec = []
lens_rec = []
bar_lens_rec = []

if __name__ == '__main__':
  midis = os.listdir(midi_dir)
  # print (len(midis))
  tmp = 0
  for mf in midis:
    print ('>> now at', mf)
    midi_obj = miditoolkit.midi.MidiFile(
                  os.path.join(midi_dir, mf)
                )
    tempo_ev = np.argmin( np.abs(TEMPOS - midi_obj.tempo_changes[0].tempo) )
    tempo_ev = TEMPOS[ tempo_ev ]
    print (tempo_ev)
    chord_event = Dechorder.dechord(midi_obj)
    chord_marker = Dechorder.enchord(midi_obj).markers
    midi_obj.instruments[0].notes = sorted(midi_obj.instruments[0].notes, key=lambda x : x.start)
    n_bars = midi_obj.instruments[0].notes[-1].start // (4 * DEFAULT_TICKS_PER_BEAT)
    n_bars += 1
    print (n_bars)
    notes = [[{'name': 'Bar', 'value': None}] for _ in range(n_bars)]
# 4/4 operation
    flag1 = 0
    flag2 = 0
    for n in midi_obj.instruments[0].notes:
      pitches_rec.append( n.pitch )
      bar = n.start // (4 * DEFAULT_TICKS_PER_BEAT)
      # print (n.start, bar)
      duration = min(n.end - n.start, 4 * DEFAULT_TICKS_PER_BEAT)
      # print (' --', bar, position, duration)
      if flag1 < len(chord_marker) and chord_marker[flag1].time <= n.start:
        #print(chord_marker[flag1].time, n.start)
        if chord_event[flag2].quality == None and chord_event[flag2].root_pc == None:
          tmp += 1
          notes[bar].append({'name': 'Chord', 'value': 'N_N'})
        else:
          notes[bar].append({'name': 'Chord', 'value': '{}_{}'.format(chord_event[flag2].root_pc, chord_event[flag2].quality)})
        while flag2 < len(chord_event)-1 and chord_event[flag2] == chord_event[flag2+1]:
            flag2 += 1
        flag2 += 1
        flag1 += 1
      if n.velocity > 127:
        n.velocity -= 127
        position = ( n.start % (4 * DEFAULT_TICKS_PER_BEAT) ) // 160
        if position % 3 != 0:
          notes[ bar ].append({'name': 'Beat', 'value': '12_{}'.format(position)})
        else:
          notes[ bar ].append({'name': 'Beat', 'value': '16_{}'.format(position)})
      else:
        position = ( n.start % (4 * DEFAULT_TICKS_PER_BEAT) ) // 120
        notes[ bar ].append({'name': 'Beat', 'value': '16_{}'.format(position)})
        
      notes[ bar ].append({'name': 'Note_Pitch', 'value': n.pitch})
      notes[ bar ].append({'name': 'Note_Duration', 'value': duration})
      notes[ bar ].append({'name': 'Note_Velocity', 'value': n.velocity})
# to ensure everything is correct(more than 1 attrs)
    for bar in range(n_bars):
      if len(notes[bar]) > 1:
        notes = notes[ bar : ]
        break
    # print (bar)
    n_bars_rec.append(len(notes))

    piece_evs = [{'name': 'Tempo', 'value': tempo_ev}] + \
                list(chain(*notes)) + \
                [{'name': 'EOS', 'value': None}]
    lens_rec.append(len(piece_evs))

    bar_pos = np.where(
      np.array([x['name'] for x in piece_evs]) == 'Bar'
    )[0]
    print (bar_pos)

    pickle.dump(
       (bar_pos.tolist(), piece_evs),
       open(os.path.join(out_dir, mf.split('_')[0] + '.pkl'), 'wb'),
       protocol=pickle.HIGHEST_PROTOCOL
    )

    print("done!")
    bar_lens_rec.extend(
      (np.array(bar_pos.tolist()[1:] + [len(piece_evs) - 1]) - bar_pos).tolist()
    )
  print("all finished!")
  print(tmp)
  # plt.clf()
  # plt.hist(pitches_rec, bins=50, rwidth=0.8)
  # plt.title('Distribution of Pitches')
  # plt.tight_layout()
  # plt.savefig('exp_data_analysis/note_pitches.jpg')

  # plt.clf()
  # plt.hist(n_bars_rec, bins=50, rwidth=0.8)
  # plt.title('Distribution of # Bars')
  # plt.tight_layout()
  # plt.savefig('exp_data_analysis/n_bars.jpg')

  # plt.clf()
  # plt.hist(lens_rec, bins=50, rwidth=0.8)
  # plt.title('Distribution of # Events')
  # plt.tight_layout()
  # plt.savefig('exp_data_analysis/n_events.jpg')

  # plt.clf()
  # plt.hist(bar_lens_rec, bins=50, rwidth=0.8)
  # plt.title('Distribution of # Events per Bar')
  # plt.tight_layout()
  # plt.savefig('exp_data_analysis/n_bar_events.jpg')

  #print (sorted(lens_rec, reverse=True)[:10])
  #print (min(pitches_rec), max(pitches_rec))
  #print (np.mean(bar_lens_rec))