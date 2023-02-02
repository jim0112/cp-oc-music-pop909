import os, pickle, random, copy
import numpy as np
import miditoolkit
import torch
from inference import generate_fast
##############################
# constants
##############################
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16
class NoteEvent(object):
  def __init__(self, pitch, bar, position, duration, velocity):
    self.pitch = pitch
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
    self.duration = duration
    self.velocity = velocity
def event_to_midi(events, output_midi_path=None, is_full_event=False, 
                  enforce_tempo=False, enforce_tempo_val=None, return_tempo=False):
  temp_notes = []
  temp_tempos = []
  for ev in events:
    pitch = ev[0]
    vel = ev[1]
    dur = ev[2]
    pos = ev[3]
    bar = ev[4]
    temp_notes.append(NoteEvent(pitch + 20, bar - 1, pos - 1, dur * 120, vel * 2 + 1))
  midi_obj = miditoolkit.midi.parser.MidiFile()
  midi_obj.instruments = [
    miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
  ]
  for n in temp_notes:
    midi_obj.instruments[0].notes.append(
      miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
    )
  if enforce_tempo is False:
    for t in temp_tempos:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )
  else:
    if enforce_tempo_val is None:
      enforce_tempo_val = temp_tempos[1]
    for t in enforce_tempo_val:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )
  if output_midi_path is not None:
    midi_obj.dump(output_midi_path)
  if not return_tempo:
    return midi_obj
  else:
    return midi_obj, temp_tempos

if __name__ == "__main__":
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = MusicTransformer()
  model.load_state_dict(torch.load("result_octuple/param/ep640_loss0.809_params.pt"))
  model.eval()
  f = "pop909_vocab.pkl"
  event2idx = pickle.load(open(f, 'rb'))[0]
  idx2event = pickle.load(open(f, 'rb'))[1]
  event , _ = generate_fast(model)
  midi_obj = event_to_midi(event, "generated_music.mid")