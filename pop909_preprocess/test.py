from beat_align import DEFAULT_RESOLUTION, DEFAULT_TICKS_PER_BEAT
import miditoolkit
from miditoolkit import midi
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from itertools import chain
from chorder import Chord, Dechorder
import pickle
import random

def convert_event(event_seq, event2idx, to_ndarr=True):
  if isinstance(event_seq[0], dict):
    event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
  else:
    event_seq = [event2idx[e] for e in event_seq]

  if to_ndarr:
    return np.array(event_seq)
  else:
    return event_seq
notes = [[{'name' : 'Bar', 'value' : None}] for _ in range(5)]
#Notes = [3 for _ in range(5)]
#print(Notes)
for i in range(5):
    notes[i].append({'name': 'Beat', 'value': 10})
print(notes)
notes = list(chain(*notes))
print(notes[:-1])
#print(notes[1]['name'])
piece = np.array([x['name'] for x in notes])
print(piece)
pieces = np.where(piece == 'Bar')[0]
print(pieces)
print((pieces.tolist(), notes))
print('-' * 20)
## make midi
#midi_obj = miditoolkit.midi.MidiFile()
#midi_obj.time_signature_changes = [
#    miditoolkit.midi.containers.TimeSignature(4, 4, 0)
#]
#midi_obj.tempo_changes = [
#    miditoolkit.midi.containers.TempoChange(bpm, 0)
#]
#midi_obj.instruments = [miditoolkit.midi.Instrument(0, name='piano_melody')]
#for n in notes:
#  midi_obj.instruments[0].notes.append(
#    miditoolkit.midi.containers.Note(
#        n['velocity'], n['pitch'], n['st_tick'], n['st_tick'] + n['dur_tick']
#    )
#  )
#midi_obj.dump(path)
## miditoolkit usage
#midi_obj = miditoolkit.midi.MidiFile(os.path.join(root_dir, pdir))
#midi_obj.instrument[0].notes #(4 attribute: start, end, velocity, pitch)
#midi_obj.tempo_changes #(temple, time)
#midi_obj.tick_per_beat

# testing array slicing

a = np.random.randint(1,30,(3,3,3))
print(a)
print('-' * 20)
print(a[1, :, :])
print(a[1])
print('-' * 20)
print(a[:, -1, :])
print(a[:, -1])
print('-' * 20)
print(a[:, :, 1])
print('-' * 20)

#midi_obj = miditoolkit.midi.MidiFile("POP909/001/001.mid")
#print(len(midi_obj.instruments[0].notes))
#print(len(midi_obj.instruments[1].notes))
#print(len(midi_obj.instruments[2].notes))

mask = np.ones((5, 4), dtype=bool)
print(mask)
mask[:, :2] = False
print(mask)
qq = np.zeros((5,))
print(qq)
qq = qq.tolist()
qq.extend([7 for _ in range(5)])
print(qq)

new = np.array([1,2,3,4,5])
probs = np.exp(new / 2) / np.sum(np.exp(new / 2))
index_probs = np.argsort(probs)[::-1]
probs = probs[::-1]
print(probs, index_probs)
cusum_probs = np.cumsum(probs)
print(cusum_probs)
after_threshold = cusum_probs > 0.5
last_index = np.where(after_threshold)[0]
print(last_index)
word = np.random.choice(index_probs, size=1, p=probs)[0]
print(word)
####### testing vocab and data #######
f = "pop909_vocab.pkl"
event2idx = pickle.load(open(f, 'rb'))[0]
idx2event = pickle.load(open(f, 'rb'))[1]
src_dir = "new_pop909_with_bars"
print(len(event2idx))
#piece1_dir = src_dir + '/909.pkl'
#piece = pickle.load(open(piece1_dir, 'rb'))[1]
#print(piece)
#for p in piece:
#    if p['name'] == 'Chord' and p['value'] == 'N_N':
#        print("哭啊真的有")
#        break

generated_music = "../music_generated/new_generated_music48/generated_music"
tmp = 0
rec = []
#for i in range(6):
#  print("inter-onset interval {}".format(i+1))
#  name = generated_music + str(i+1) + ".mid"
#  midi_obj = miditoolkit.midi.MidiFile(name)
#  midi_obj.instruments[0].notes = sorted(midi_obj.instruments[0].notes, key=lambda x : x.start)
#  for index, n in enumerate(midi_obj.instruments[0].notes):
#    if index != 0:
#      if n.start - tmp != 0:
#        rec.append(n.start - tmp)
#    tmp = n.start
#print("done!")
#plt.clf()
#plt.xlim((0,480*4))
#plt.hist(rec, bins=100, rwidth=0.3)
#plt.title('Inter-onset Interval')
#plt.tight_layout()
#plt.savefig('data_analysis/Inter-onset_realdata.jpg')
f = 'new_pop909_with_bars'
test = pickle.load(open(os.path.join(f, '805.pkl'), 'rb'))
print(test)

aa = random.random() > 0.05
print(aa)