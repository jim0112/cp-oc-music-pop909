import miditoolkit
import os, pickle
from copy import deepcopy
import numpy as np
from exploratory import read_info_file

from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt

DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_RESOLUTION = 120

root_dir = 'POP909'
melody_out_dir = 'pop909_melody_midi'

downbeat_records = []
downbeat_scores = []
all_bpms = []

def justify_tick(n_beats):
  n_ticks = n_beats * DEFAULT_TICKS_PER_BEAT
  return int(DEFAULT_RESOLUTION * round(n_ticks / DEFAULT_RESOLUTION))

def bpm2sec(bpm):
  return 60. / bpm

def calc_accum_secs(bpm, n_ticks, ticks_per_beat):
  return bpm2sec(bpm) * n_ticks / ticks_per_beat
# not used
def find_downbeat_idx(minor_dbt, major_dbt, n_beats=4):
  matched_idx, max_score = -1, -1

  for st_idx in range(n_beats):
    minor_cand = minor_dbt[ st_idx::n_beats ]
    major_cand = major_dbt[ st_idx::n_beats ]
    # print (minor_cand[:4], major_cand[:4])
    
    score = np.dot(minor_cand, major_cand)
    # score = sum(minor_cand + major_cand)
    if score > max_score:
      matched_idx = st_idx
      max_score = score

  print ('[dbeat align] idx = {} (score = {:.2f} %)'.format(
    matched_idx, 100 * max_score / len(minor_dbt)
  ))

  return matched_idx, max_score / len(minor_dbt)
# clear
def find_downbeat_idx_audio(audio_dbt):
  for st_idx in range(4):
    if audio_dbt[ st_idx ] == 1.:
      return st_idx
# clear
def get_note_time_sec(note, tempo_bpms, ticks_per_beat,
                      tempo_change_ticks, tempo_accum_times
                      ):
  st_seg = np.searchsorted(tempo_change_ticks, note.start, side='left') - 1
  ed_seg = np.searchsorted(tempo_change_ticks, note.end, side='left') - 1
  # print (note.start, tempo_change_ticks[ st_seg ])

  start_sec = tempo_accum_times[ st_seg ] +\
              calc_accum_secs(
                tempo_bpms[ st_seg ],
                note.start - tempo_change_ticks[ st_seg ],
                ticks_per_beat
              )
  end_sec = tempo_accum_times[ ed_seg ] +\
            calc_accum_secs(
              tempo_bpms[ ed_seg ],
              note.end - tempo_change_ticks[ ed_seg ],
              ticks_per_beat
            )

  return start_sec, end_sec            
# clear
def align_notes_to_secs(midi_obj):
  tempo_bpms = []
  tempo_change_ticks = []
  tempo_accum_times = []
  for tc in midi_obj.tempo_changes:
    # print (tc.tempo, tc.time)
    if tc.time == 0:
      tempo_accum_times.append( 0. )
    else:
      tempo_accum_times.append(
        tempo_accum_times[-1] + \
        calc_accum_secs(
          tempo_bpms[-1], 
          tc.time - tempo_change_ticks[-1], 
          midi_obj.ticks_per_beat
        )
      )

    tempo_bpms.append(tc.tempo)
    tempo_change_ticks.append(tc.time)

  # print (tempo_accum_times)
  # print (midi_obj.instruments[0].notes[0].start, midi_obj.instruments[1].notes[0].start)

  vocal_notes = []
  for note in midi_obj.instruments[0].notes:
    note_st_sec, note_ed_sec = get_note_time_sec(
                                note, tempo_bpms,
                                midi_obj.ticks_per_beat, tempo_change_ticks,
                                tempo_accum_times
                              )
    # print (note_st_sec, note_ed_sec)
    vocal_notes.append(
      {'st_sec': note_st_sec, 'ed_sec': note_ed_sec, 'pitch': note.pitch, 'velocity': note.velocity}
    )

  piano_notes = []
  for note in midi_obj.instruments[1].notes:
    note_st_sec, note_ed_sec = get_note_time_sec(
                                note, tempo_bpms,
                                midi_obj.ticks_per_beat, tempo_change_ticks,
                                tempo_accum_times
                              )
    # print (note_st_sec, note_ed_sec)
    piano_notes.append(
      {'st_sec': note_st_sec, 'ed_sec': note_ed_sec, 'pitch': note.pitch, 'velocity': note.velocity}
    )

  return vocal_notes, piano_notes

def group_notes_per_beat(notes, beat_times):
  n_beats = len(beat_times)
  note_groups = [[] for _ in range(n_beats)]
  cur_beat = 0

  # violation_front, violation_back = [], []
  notes = sorted(notes, key=lambda x: (x['st_sec'], -x['pitch']))
    
  for note in notes:
    while cur_beat < (n_beats - 1) and note['st_sec'] > beat_times[ cur_beat + 1 ]:
      # print (cur_beat, note['st_sec'], beat_times[ cur_beat + 1 ]) 
      cur_beat += 1

    if cur_beat == 0 and note['st_sec'] < beat_times[0]:
      if note['st_sec'] >= (beat_times[0] - 0.1) and note['ed_sec'] - note['st_sec'] > 0.2:
        note['st_sec'] = beat_times[0]
      else:
        # print ('[violation] {:.2f} < {:.2f}'.format(
        #   note['st_sec'], beat_times[0]
        # ))
        # vio_beat_count = math.ceil(
        #                   (beat_times[0] - note['st_sec']) / (beat_times[1] - beat_times[0])
        #                 )
        # violation_front.append(
        #   (vio_beat_count, note)
        # )
        continue

    if cur_beat == n_beats - 1:
      if note['st_sec'] - beat_times[-1] > beat_times[-1] - beat_times[-2]:
        # print ('[violation] {:.2f} - {:.2f} - {:.2f}'.format(
        #     note['st_sec'], beat_times[-1], beat_times[-2]
        # ))
        # vio_beat_count = (note['st_sec'] - beat_times[-1]) // (beat_times[-1] - beat_times[-2])
        # violation_back.append(
        #   (vio_beat_count, note)
        # )
        continue

    note_groups[ cur_beat ].append( deepcopy(note) )

  # return note_groups, violation_front, violation_back
  return note_groups
# clear and may not use
def remove_piano_notes_collision(vocal_notes, piano_notes):
  n_beats = len(vocal_notes)

  for beat in range(n_beats):
    if (beat - 1 >= 0 and len(vocal_notes[ beat - 1 ])) or \
       len (vocal_notes[ beat ]) or \
       (beat + 1 < n_beats and len(vocal_notes[ beat + 1 ])) or \
       (beat + 2 < n_beats and len(vocal_notes[ beat + 2 ])):
      piano_notes[ beat ] = []

  return piano_notes
# generate tick information (ticks are what really matter in miditoolkit)
# clear
def quantize_notes(notes, beat_times, downbeat_idx):
  quantized = [[] for _ in range(len(beat_times))]

  if downbeat_idx == 1:
    cur_tick = 3 * DEFAULT_TICKS_PER_BEAT
  elif downbeat_idx == 2:
    cur_tick = 2 * DEFAULT_TICKS_PER_BEAT
  elif downbeat_idx == 3:
    cur_tick = DEFAULT_TICKS_PER_BEAT
  else:
    cur_tick = 0

  for b_idx, beat_notes in enumerate(notes):
    beat_dur = beat_times[b_idx + 1] - beat_times[b_idx]\
                  if b_idx < len(notes) - 1 else beat_times[-1] - beat_times[-2]
    beat_st_sec = beat_times[b_idx]
  
    for note in beat_notes:
      note_dur_tick = justify_tick( (note['ed_sec'] - note['st_sec']) / beat_dur )
      if note_dur_tick == 0:
        continue
      note_st_tick = cur_tick +\
                     justify_tick( (note['st_sec'] - beat_st_sec) / beat_dur )

      if note_st_tick < 0:
        print (note['st_sec'], beat_st_sec, b_idx, cur_tick)
        print ('[violation]', note_st_tick)

      note['st_tick'] = note_st_tick
      note['dur_tick'] = note_dur_tick
      quantized[ b_idx ].append( deepcopy(note) )

    cur_tick += DEFAULT_TICKS_PER_BEAT

  return quantized
# clear
def merge_and_resolve_polyphony(vocal_notes, piano_notes):
  vocal_notes = list(chain(*vocal_notes))
  piano_notes = list(chain(*piano_notes))

  notes = sorted(
            vocal_notes + piano_notes, 
            key=lambda x : (x['st_tick'], -x['pitch'])
          )
  # for i, n in enumerate(notes):
  #   if i != 0 and n['st_tick'] == notes[i - 1]['st_tick']:
  #     print (n['pitch'], notes[i - 1]['pitch'])

  final_notes = []
  cur_tick = -1
  for n in notes:
    if n['st_tick'] == cur_tick:
      continue
    else:
      cur_tick = n['st_tick']
      final_notes.append(n)

  return final_notes
# clear
def dump_melody_midi(notes, bpm, midi_out_path):
  midi_obj = miditoolkit.midi.MidiFile()
  midi_obj.time_signature_changes = [
    miditoolkit.midi.containers.TimeSignature(4, 4, 0)
  ]
  midi_obj.tempo_changes = [
    miditoolkit.midi.containers.TempoChange(bpm, 0)
  ]
  midi_obj.instruments = [miditoolkit.midi.Instrument(0, name='piano_melody')]

  for n in notes:
    midi_obj.instruments[0].notes.append(
       miditoolkit.midi.containers.Note(
         n['velocity'], n['pitch'], n['st_tick'], n['st_tick'] + n['dur_tick']
       )
    )

  midi_obj.dump(midi_out_path)
  return

# most important part
def align_midi_beats(piece_dir):
  midi_beat_path = os.path.join(piece_dir, 'beat_midi.txt')
  audio_beat_path = os.path.join(piece_dir, 'beat_audio.txt')
  # midi_beat_times = read_info_file(midi_beat_path, [0])[0]
  midi_beat_times = read_info_file(audio_beat_path, [0])[0]
  midi_beat_idx = read_info_file(audio_beat_path, [1])[1]
  # midi_beats = read_info_file(midi_beat_path, [1, 2])
  # midi_beats_minor, midi_beats_major = midi_beats[1], midi_beats[2]
  
  # downbeat_idx, _ = find_downbeat_idx(midi_beats_minor, midi_beats_major)
  # downbeat_records.append( downbeat_idx )
  # downbeat_scores.append( _ )
  downbeat_idx = find_downbeat_idx_audio(midi_beat_idx)
  print (downbeat_idx)
  ######################################################################
  midi_obj = miditoolkit.midi.MidiFile(
              os.path.join(root_dir, pdir, pdir + '.mid')
            )

  vocal_notes, piano_notes = align_notes_to_secs(midi_obj)
  vocal_notes = group_notes_per_beat(vocal_notes, midi_beat_times)
  piano_notes = group_notes_per_beat(piano_notes, midi_beat_times)
  # print (vocal_notes[:4])
  # print (piano_notes[:4])
  # if vocal_vio_fr or vocal_vio_bk or piano_vio_fr or piano_vio_bk:
  #   print ('<< violated >>', vocal_vio_fr, vocal_vio_bk, piano_vio_fr, piano_vio_bk)
  # print ('==========================')

  # piano_notes = remove_piano_notes_collision(vocal_notes, piano_notes)

  vocal_notes = quantize_notes(vocal_notes, midi_beat_times, downbeat_idx)
  piano_notes = quantize_notes(piano_notes, midi_beat_times, downbeat_idx)
  # print (piano_notes[:8])
  # print ('[# notes]', vocal_notes + piano_notes)

  final_notes = merge_and_resolve_polyphony(vocal_notes, piano_notes)
  print ('# notes:', len(final_notes))

  final_bpm = 60. / ( (midi_beat_times[-1] - midi_beat_times[0]) / (len(midi_beat_times) - 1) )
  final_bpm = np.round(final_bpm, 2)
  # print (final_bpm, midi_obj.tempo_changes[0])
  all_bpms.append( final_bpm )

  dump_melody_midi(
    final_notes,
    final_bpm,
    os.path.join(melody_out_dir, piece_dir.split('/')[-1] + '_melody.mid')
  )

# start main
if __name__ == '__main__':
  pieces_dir = pickle.load(
                  open('pop909_with_bars/qual_pieces.pkl', 'rb'), 
                )
  print (len(pieces_dir))

  for pdir in pieces_dir:
    print ('>> now at #{:03d}'.format(int(pdir)))
    align_midi_beats(
      os.path.join(root_dir, pdir)
    )

  # print (Counter(downbeat_records))
  # plt.clf()
  # plt.hist(downbeat_scores, bins=20, rwidth=0.8, range=(0, 0.28))
  # plt.title('Distribution of Downbeat Scores')
  # plt.tight_layout()
  # plt.savefig('exp_data_analysis/downbeat_scores.jpg')

  plt.clf()
  plt.hist(all_bpms, bins=20, rwidth=0.8)
  plt.title('Distribution of BPMs')
  plt.tight_layout()
  plt.savefig('exp_data_analysis/tempo_bpms.jpg')