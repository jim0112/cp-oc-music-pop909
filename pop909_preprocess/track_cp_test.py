import miditoolkit
from miditoolkit import midi
from exploratory import read_info_file
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from itertools import chain
from chorder import Chord, Dechorder
src_dir = './multitrack_pop909_melody_midi'
out_dir = './cp/cp_dataset'
root_dir = 'POP909'
VELO = np.arange(3, 128, 2)
TEMPOS = np.arange(32, 227, 3) # 32 -> 224
midis = os.listdir(src_dir)
max_bars = -1
def find_downbeat_idx_audio(audio_dbt):
  for st_idx in range(4):
    if audio_dbt[ st_idx ] == 1.:
      return st_idx
def extract_and_merge_tracks_info(midi_obj):
  vocal_notes = []
  for note in midi_obj.instruments[0].notes:
    vocal_notes.append(
      {'start': note.start, 'end': note.end, 'pitch': note.pitch, 'velocity': note.velocity, 'track' : 0}
    )
  piano_notes = []
  for note in midi_obj.instruments[1].notes:
    piano_notes.append(
      {'start': note.start, 'end': note.end, 'pitch': note.pitch, 'velocity': note.velocity, 'track' : 1}
    )
  accompany_notes = []
  for note in midi_obj.instruments[2].notes:
    accompany_notes.append(
      {'start': note.start, 'end': note.end, 'pitch': note.pitch, 'velocity': note.velocity, 'track' : 2}
    )
  notes = sorted(
            vocal_notes + piano_notes + accompany_notes, 
            key=lambda x : (x['start'], -x['pitch'])
          )
  final_notes = []
  for n in notes:
    final_notes.append(n)

  return final_notes

for piece in midis:
  print ('>> now at', piece)
  midi_obj = miditoolkit.midi.MidiFile(os.path.join(src_dir, piece))
  chord_event = Dechorder.dechord(midi_obj)
  chord_marker = Dechorder.enchord(midi_obj).markers
  notes = extract_and_merge_tracks_info(midi_obj)
  x = []
  bar_st_rec = []
  pre_bar = -1
  pre_tempo = -1
  pre_pos = -1
  pre_chord = ''
  flag1 = 0
  flag2 = 0
  chord = 'IGN'
  number = piece.split('_')[0]
  # for tempo
  audio_beat_path = os.path.join(root_dir, number, 'beat_audio.txt')
  midi_beat_times = read_info_file(audio_beat_path, [0])[0]
  midi_beat_index = read_info_file(audio_beat_path, [1])[1]
  downbeat_idx = find_downbeat_idx_audio(midi_beat_index)
  if downbeat_idx == 1:
    cur_tick = 3 * 480
  elif downbeat_idx == 2:
    cur_tick = 2 * 480
  elif downbeat_idx == 3:
    cur_tick = 480
  else:
    cur_tick = 0
  start_index = 0
  index = 0
  #[chord, tempo, barbeat, type, track, pitch, dur, vel]
  for n in notes:
    #chord
    if flag1 < len(chord_marker) and chord_marker[flag1].time <= n['start']:
      #print(chord_marker[flag1].time, n['start]])
      if chord_event[flag2].quality == None and chord_event[flag2].root_pc == None:
        chord = 'N_N'
      else:
        chord = '{}_{}'.format(chord_event[flag2].root_pc, chord_event[flag2].quality)
      while flag2 < len(chord_event)-1 and chord_event[flag2] == chord_event[flag2+1]:
        flag2 += 1
      flag2 += 1
      flag1 += 1
    # tempo
    while n['start'] >= cur_tick + 480:
      cur_tick += 480
      start_index += 1
    if n['start'] >= cur_tick and n['start'] < cur_tick + 480:
      # belongs to this beat
      if start_index >= len(midi_beat_times) - 1:
        tempo = 60 / (midi_beat_times[-1] - midi_beat_times[-2])
      else:
        tempo = 60 / (midi_beat_times[start_index + 1] - midi_beat_times[start_index])
    tempo_index = np.argmin(np.abs(TEMPOS - tempo))
    tempo = TEMPOS[tempo_index]
    # pos
    pos = ( n['start'] % (4 * 480) ) // 120
    # bar
    bar = n['start'] // (4 * 480)
    if bar != pre_bar:
      x.append([0, 0, 17, 1, 3, 0, 0, 0])
      pre_bar = bar
      bar_st_rec.append(index)
      index += 1
    if pos != pre_pos:
      if chord == pre_chord:
        fill_chord = 'conti'
      else:
        fill_chord = chord
        pre_chord = chord
      if tempo == pre_tempo:
        fill_tempo = 1 # conti
      else:
        fill_tempo = int((tempo - 26) / 3) # from 2 on
        pre_tempo = tempo
      x.append([fill_chord, fill_tempo, pos + 1, 1, 0, 0, 0, 0])
      pre_pos = pos
      index += 1
    #track
    track = n['track']
    #pitch
    pitch = n['pitch']
    # dur
    duration = min(n['end'] - n['start'], 4 * 480)
    # vel
    Vel_index = np.argmin(np.abs(VELO - n['velocity']))
    vel = VELO[Vel_index]
    x.append([0, 0, 0, 2, track, pitch - 20, int(duration / 120), int((vel - 1) / 2)])
    index += 1
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
chord_vocab.append('conti')
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


