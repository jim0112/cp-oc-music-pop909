import os, pickle, random, sys
sys.path.append('./exploration/')
from glob import glob

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
#from exploration.polyphonicity import compute_polyphonicity
#from exploration.rhym_freq import get_onsets_timing

#[chord, tempo, barbeat, type, track, pitch, dur, vel]
# check
def check_extreme_pitch(raw_events):
  low, high = 88, 1
  for ev in raw_events:
      if ev[3] == 2:
        low = min(low, ev[5])
        high = max(high, ev[5])
  return low, high
#check
def split_chord_value(chord_event):
  tone = chord_event.split('_')[0]
  return tone
def transpose_events(raw_events, n_keys):
  for ev in raw_events:
    if ev[3] == 2: 
      ev[5] += n_keys
    else:
      if ev[0] == 0 or 'N_N' or 'conti' or 'IGN':
        continue
      else:
        orig_tone_idx = split_chord_value(ev[0])
        new_tone_idx = (int(orig_tone_idx) + 12 + n_keys) % 12
        ev[0] = ev[0].replace('{}_'.format(orig_tone_idx), '{}_'.format(new_tone_idx)
  )
  return raw_events
# check
def pickle_load(path):
  return pickle.load(open(path, 'rb'))
def convert_event(events, event2idx):
  for ev in events:
    if ev[0] != 0:
      ev[0] = event2idx.get(ev[0])
  return events
class REMIFullSongTransformerDataset(Dataset):
  def __init__(self, data_dir, vocab_file, 
               model_enc_seqlen=128, model_dec_seqlen=3840, model_max_bars=32,
               pieces=[], do_augment=True, augment_range=range(-6, 7), 
               min_pitch=1, max_pitch=88, pad_to_same=False, use_attr_cls=True,
               appoint_st_bar=None, dec_end_pad_value=None):
    self.chord_vocab_file = vocab_file
    self.chord_event2idx = pickle_load(vocab_file)[0]
    self.chord_idx2event = pickle_load(vocab_file)[1]
    self.vocab_len = [134, 18, 3, 89, 17, 64]
    self.data_dir = data_dir ##
    self.pieces = pieces ##
    self.build_dataset() ##

    self.model_enc_seqlen = model_enc_seqlen
    self.model_dec_seqlen = model_dec_seqlen ##512
    self.model_max_bars = model_max_bars ##16

    self.do_augment = do_augment ##
    self.augment_range = augment_range ##
    self.min_pitch, self.max_pitch = min_pitch, max_pitch ## 
    self.pad_to_same = pad_to_same ##
    self.use_attr_cls = use_attr_cls
    self.dec_end_pad_value = dec_end_pad_value
    self.appoint_st_bar = appoint_st_bar ##
    if self.dec_end_pad_value == None:
      self.pad_token = 0
    else:
      self.pad_token = self.dec_end_pad_value
# check
  def build_dataset(self):
    if not any(self.pieces):
      self.pieces = sorted( glob(os.path.join(self.data_dir, '*.pkl')) )
    else:
      self.pieces = sorted( [os.path.join(self.data_dir, p) for p in self.pieces] )

    self.bar_pos_rec = []
    for i, p in enumerate(self.pieces):
      bar_pos, p_evs = pickle_load(p)
      if not i % 200:
        print ('[preparing data] now at #{}'.format(i))
      bar_pos.append(len(p_evs))
      self.bar_pos_rec.append(bar_pos)
# check
# chord, bar-beat, type, pitch, duration, velocity
# type : 0-IGN 1-metrix 2-note
  def get_sample_from_file(self, piece_idx):
    piece_evs = pickle_load(self.pieces[piece_idx])[1]
    piece_bar_pos = self.bar_pos_rec[piece_idx]
    if piece_bar_pos[-1] <= self.model_dec_seqlen:
      picked_st_bar = 0
      picked_end_bar = -1
      piece_evs = piece_evs[piece_bar_pos[picked_st_bar] : piece_bar_pos[picked_end_bar]]
    else:
      index = 0
      while(piece_bar_pos[index] + self.model_dec_seqlen  <= piece_bar_pos[-1]):
        index += 1
      picked_st_bar = random.choice(range(index))
      picked_end_bar = 0
      for i in range(picked_st_bar, len(piece_bar_pos)):
        if piece_bar_pos[i] - piece_bar_pos[picked_st_bar] > self.model_dec_seqlen:
          picked_end_bar = i - 1
          break
        else:
          picked_end_bar = -1
      piece_evs = piece_evs[ piece_bar_pos[picked_st_bar] : piece_bar_pos[picked_end_bar]]
    return piece_evs
# check
# [chord, tempo, barbeat, type, track, pitch, dur, vel]
  def pad_sequence(self, seq, maxlen, pad_value=None):
    tmp = [0, 0, 0, 0, 3, 0, 0, 0] # track : 3 -> IGN
    for i in range(maxlen - len(seq)):
      seq.append(tmp)
    return seq
# used 轉調？
  def pitch_augment(self, bar_events):
    bar_min_pitch, bar_max_pitch = check_extreme_pitch(bar_events)
    n_keys = random.choice(self.augment_range)
    while bar_min_pitch + n_keys < self.min_pitch or bar_max_pitch + n_keys > self.max_pitch:
      n_keys = random.choice(self.augment_range)
    augmented_bar_events = transpose_events(bar_events, n_keys) ##
    return augmented_bar_events
  def __len__(self):
    return len(self.pieces)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    bar_tokens = self.get_sample_from_file(idx) ##
    if self.do_augment:
      bar_tokens = self.pitch_augment(bar_tokens) ##
    #print(bar_tokens)
    # print (bar_pos)
    # print ('[no. {:06d}] len: {:4} | last: {}'.format(idx, len(bar_tokens), self.idx2event[ bar_tokens[-1] ]))
    bar_tokens = convert_event(bar_tokens, self.chord_event2idx)
    if self.pad_to_same: ##
      inp = self.pad_sequence(bar_tokens, self.model_dec_seqlen + 1) ##
    #important
    target = np.array(inp[1:], dtype=int) ##
    inp = np.array(inp[:-1], dtype=int) ##
    assert len(inp) == len(target)

    return {
      'length': min(len(bar_tokens), self.model_dec_seqlen),
      'id': idx,
      'dec_input': inp[:self.model_dec_seqlen],
      'dec_target': target[:self.model_dec_seqlen]
    }

if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    './remi_dataset', './pickles/remi_wenyi_vocab.pkl', do_augment=False, use_attr_cls=False,
    model_max_bars=8, model_dec_seqlen=1536, model_enc_seqlen=192, min_pitch=22, max_pitch=105
  )
  print (dset.bar_token, dset.pad_token, dset.vocab_size)
  print ('length:', len(dset))

  # for i in random.sample(range(len(dset)), 100):
  for i in range(len(dset)):
    sample = dset[i]
    # print (i, len(sample['bar_pos']), sample['bar_pos'])
    print (i)
    print ('******* ----------- *******')
    print (sample['dec_input'][:16])
    print (sample['dec_target'][:16])
    # print (sample['enc_padding_mask'][:32, :16])
    # print (sample['length'])


  # rfreq_cnt = [0 for x in range(8)]
  # polyph_cnt = [0 for x in range(8)]
  # # for i in range(len(dset)):
  # for i in random.sample(range(len(dset)), 100):
  #   # if not i % 1000:
  #   #   print ('>> sample {}'.format(i))
  #   sample = dset[i]
  #   # print (sample['length'], sample['input'].shape)
  #   # print ('length:', sample['length'])
  #   # print ('poly: {}, rfreq: {}'.format(sample['polyph_cls'], sample['rhymfreq_cls']))
  #   print (sample['polyph_cls'][:sample['length']])
  #   print (sample['rhymfreq_cls'][:sample['length']])

    # for b in range(4):
    #   polyph_cnt[ sample['polyph_cls'][b] ] += 1
    #   rfreq_cnt[ sample['rhymfreq_cls'][b] ] += 1
    # print ('mask:', sample['emb_mask'][:64])

  # print (polyph_cnt)
  # print (rfreq_cnt)

  # dloader = DataLoader(dset, batch_size=4, shuffle=False, num_workers=24)
  # for i, batch in enumerate(dloader):
  #   for k, v in batch.items():
  #     if torch.is_tensor(v):
  #       print (k, ':', v.dtype, v.size())
