import os, pickle, random, copy
import numpy as np
import miditoolkit
import torch
from cp_music_transformer import MusicTransformer
from cp_inference import generate_fast
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
  cur_bar = 0
  cur_beat = 0
  temp_notes = []
  for i in range(len(events)):
    barbeat = events[i][1]
    typee = events[i][2]
    pitch = events[i][3]
    dur = events[i][4]
    vel = events[i][5]
    if typee == 1 and barbeat == 17:
      cur_bar += 1
    elif typee == 1 and barbeat != 17:
      cur_beat = barbeat
    elif typee == 2:
      temp_notes.append(NoteEvent(pitch + 20, cur_bar - 1, cur_beat - 1, dur * 120, vel * 2 + 1))
  midi_obj = miditoolkit.midi.parser.MidiFile()
  midi_obj.instruments = [
    miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
  ]
  for n in temp_notes:
    midi_obj.instruments[0].notes.append(
      miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
    )
  if output_midi_path is not None:
    midi_obj.dump(output_midi_path)
  return midi_obj

if __name__ == "__main__":
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  n_token = [134, 67, 18, 3, 4, 89, 17, 64]
  model = MusicTransformer(
    n_token, 12, 8, 256, 1024, 256
  ).to(device)
  model.load_state_dict(torch.load("result_new/params/ep870_loss0.459_params.pt"))
  model.eval()
  event = generate_fast(model)
  midi_obj = event_to_midi(event, "generated_music.mid")
