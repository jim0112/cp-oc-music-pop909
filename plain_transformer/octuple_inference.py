#######################################
# inference.py
#######################################
import torch
import numpy as np
import time
from scipy.stats import entropy
from utils import numpy_to_tensor, tensor_to_numpy
#temp, top_p = 1.2, 0.9
temp, top_p = 1.0, 0.9
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
########################################
# generation
########################################
def get_position_idx(event):
  return int(event.split('_')[-1])
def generate_fast(model, max_events=512, max_bars=8, skip_check=False, gpuid=None):
  cur_bar = 0
  generated = []
  target_bars, generated_bars = max_bars, 0
  steps = 0
  time_st = time.time()
  cur_pos = 1
  failed_cnt = 0
  entropies = []
  while generated_bars < target_bars:
    dec_input = numpy_to_tensor([generated]).long()
    # print (dec_input.size(), dec_seg_emb.size())
    # sampling
    pitch, vel, dur, pos, bar = model(dec_input, keep_last_only=True)
    pitch = tensor_to_numpy(pitch[0])
    vel = tensor_to_numpy(vel[0])
    dur = tensor_to_numpy(dur[0])
    pos = tensor_to_numpy(pos[0])
    bar = tensor_to_numpy(bar[0])
    # print (logits.shape)
    # softmax 
    probs_pitch = temperature(pitch, temp)
    probs_vel = temperature(vel, temp)
    probs_dur = temperature(dur, temp)
    probs_pos = temperature(pos, temp)
    probs_bar = temperature(bar, temp)
    # choose one with high prob
    word_pitch = nucleus(probs_pitch, top_p)
    word_vel = nucleus(probs_vel, top_p)
    word_dur = nucleus(probs_dur, top_p)
    word_pos = nucleus(probs_pos, top_p)
    word_bar = nucleus(probs_bar, top_p)
    if not skip_check:
      if not word_pos >= cur_pos:
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 256:
          print ('[FATAL] model stuck, exiting with generated events ...')
          return generated
        continue
      else:
        cur_pos = word_pos
        failed_cnt = 0
    if word_bar != cur_bar:
      generated_bars += 1
      cur_pos = 1
      cur_bar = word_bar
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))
    generated.append([word_pitch, word_vel, word_dur, word_pos, word_bar])

    steps += 1
    if len(generated) > max_events:
      print ('[info] max events reached')
      break
  print ('-- generated events:', len(generated))
  print ('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
  print ('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
  return generated[:-1]