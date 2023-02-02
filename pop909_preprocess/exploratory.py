import miditoolkit
import os, pickle
import matplotlib.pyplot as plt
from collections import Counter

root_dir = 'POP909'

def read_info_file(fpath, tgt_cols):
  with open(fpath, 'r') as f:
    lines = f.read().splitlines()

  ret_dict = {col: [] for col in tgt_cols}
  for l in lines:
    l = l.split()
    for col in tgt_cols:
      ret_dict[col].append( float(l[col]) )

  return ret_dict

if __name__ == '__main__':
  pieces_dir = [ i for i in os.listdir(root_dir) 
                  if os.path.isdir( os.path.join(root_dir, i) ) 
                ]
  print (len(pieces_dir))

  # note_lens = []
  # n_audio_beats = []
  qualified_quad = 0
  qualified_triple = 0
  qualified_pieces = []
  for pdir in pieces_dir:
    # midi_obj = miditoolkit.midi.MidiFile(
    #               os.path.join(root_dir, pdir, pdir + '.mid')
    #             )
    # print ('#{}'.format(pdir))
    # print ('#{} :'.format(pdir), midi_obj.tempo_changes, '\n      ', midi_obj.time_signature_changes, '\n      ', midi_obj.instruments[0].notes[:10])

    # for note in midi_obj.instruments[0].notes + midi_obj.instruments[1].notes:
    #   note_lens.append(note.end - note.start)

    audio_beat_path = os.path.join(root_dir, pdir, 'beat_audio.txt')
    audio_beats = read_info_file(audio_beat_path, [1])[1]

    midi_beat_path = os.path.join(root_dir, pdir, 'beat_midi.txt')
    midi_beats = read_info_file(midi_beat_path, [1, 2])
    midi_beats_minor, midi_beats_major = midi_beats[1], midi_beats[2]

    print ('#{} : {:.0f} | {:.2f} | {:.2f}'.format(pdir, max(audio_beats), sum(midi_beats_major) / len(midi_beats_major), sum(midi_beats_minor) / len(midi_beats_major)))
    # n_audio_beats.append(max(audio_beats))
    if max(audio_beats) == 4.:
      try:
        assert abs(0.25 - sum(midi_beats_major) / len(midi_beats_major)) < 0.03
        qualified_quad += 1
        qualified_pieces.append(pdir)
      except:
        print ('[error] 4-beat !!')
    elif max(audio_beats) == 3.:
      try:
        assert abs(0.33 - sum(midi_beats_minor) / len(midi_beats_major)) < 0.03, sum(midi_beats_minor) / len(midi_beats_major)
        qualified_triple += 1
      except:
        print ('[error] 3-beat !!')

    # print (len(audio_beats), len(midi_beats_minor), len(midi_beats_major))
    # print (midi_beats_minor[:10], midi_beats_major[:10], audio_beats[:10])

  print (qualified_quad, qualified_triple)
  print (qualified_pieces[:20], len(qualified_pieces))
  pickle.dump(
    qualified_pieces, 
    open('pop909_with_bars/qual_pieces.pkl', 'wb'), 
    protocol=pickle.HIGHEST_PROTOCOL
  )
  # print (Counter(n_audio_beats))
  # print ('# total notes:', len(note_lens))
  # plt.clf()
  # plt.hist(note_lens, bins=50, rwidth=0.8, range=(0, 500))
  # plt.title('Distribution of Note Lengths (in Ticks)')
  # plt.tight_layout()
  # plt.savefig('exp_data_analysis/note_lens_short.jpg')