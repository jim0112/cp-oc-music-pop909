#######################################
# inference.py
#######################################
import torch
import numpy as np
import time
from scipy.stats import entropy
from utils import numpy_to_tensor, tensor_to_numpy
temp, top_p = 1.2, 0.9
#temp, top_p = 1.0, 0.9
########################################
# sampling utilities
########################################
# clear
def temperature(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs
def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word
def generate_fast(model, max_events=512, max_bars=10, skip_check=False, gpuid=None):
  generated = [[0,0,17,1,0,0,0]]
  target_bars, generated_bars = max_bars, 0
  steps = 0
  time_st = time.time()
  cur_pos = 1
  failed_cnt = 0
  entropies = []
  while generated_bars < target_bars: 
    dec_input = numpy_to_tensor([generated]).long()
    if len(generated) > 1:
      dec_input = dec_input.permute(1, 0, 2)
    # print (dec_input.size(), dec_seg_emb.size())
    # sampling
    chord, barbeat, typee, pitch, dur, vel  = model(dec_input, None, keep_last_only=True, inference=True)
    print(typee, typee[0])
    chord = tensor_to_numpy(chord[0])
    barbeat = tensor_to_numpy(barbeat[0])
    typee = tensor_to_numpy(typee[0])
    pitch = tensor_to_numpy(pitch[0])
    dur = tensor_to_numpy(dur[0])
    vel = tensor_to_numpy(vel[0])
    # print (logits.shape)
    # softmax 
    probs_chord = temperature(chord, temp)
    probs_barbeat = temperature(barbeat, temp)
    probs_type = temperature(typee, temp)
    probs_pitch = temperature(pitch, temp)
    probs_dur = temperature(dur, temp)
    probs_vel = temperature(vel, temp)
    # choose one with high prob
    word_chord = nucleus(probs_chord, top_p)
    word_barbeat = nucleus(probs_barbeat, top_p)
    word_type = nucleus(probs_type, top_p)
    word_pitch = nucleus(probs_pitch, top_p)
    word_dur = nucleus(probs_dur, top_p)
    word_vel = nucleus(probs_vel, top_p)
    if not skip_check:
      if word_type == 1 and word_barbeat != 17:
        if not word_barbeat >= cur_pos:
          failed_cnt += 1
          print ('[info] position not increasing, failed cnt:', failed_cnt)
          if failed_cnt >= 256:
            print ('[FATAL] model stuck, exiting with generated events ...')
            return generated
          continue
        else:
          cur_pos = word_barbeat
          failed_cnt = 0
    if word_type == 1 and word_barbeat == 17:
      generated_bars += 1
      cur_pos = 1
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))
    print('[{},{},{},{},{},{}]'.format(word_chord, word_barbeat, word_type, word_pitch, word_dur, word_vel))
    generated.append([word_chord, word_barbeat, word_type, word_pitch, word_dur, word_vel])

    steps += 1
    if len(generated) > max_events:
      print ('[info] max events reached')
      break
  print ('-- generated events:', len(generated))
  print ('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
  print ('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
  return generated[:-1]
