import pickle
import numpy as np
from itertools import chain
from chorder import Chord, Dechorder
import miditoolkit
import torch
from torch import nn
import torch.nn.functional as F
class TokenEmbedding(nn.Module):
  def __init__(self, n_token, d_embed, d_proj):
    super(TokenEmbedding, self).__init__()

    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.emb_scale = d_proj ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_embed)
    if d_proj != d_embed:
      self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
    else:
      self.emb_proj = None

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    
    if self.emb_proj is not None:
      inp_emb = self.emb_proj(inp_emb)

    return inp_emb.mul_(self.emb_scale)
def read_info_file(fpath, tgt_cols):
  with open(fpath, 'r') as f:
    lines = f.read().splitlines()

  ret_dict = {col: [] for col in tgt_cols}
  for l in lines:
    l = l.split()
    for col in tgt_cols:
      ret_dict[col].append( l[col] )

  return ret_dict
#from utils import numpy_to_tensor, tensor_to_numpy
if __name__ == "__main__":
  f = "pop909_cp_Chord_vocab.pkl"
  event2idx = pickle.load(open(f, 'rb'))[0]
  idx2event = pickle.load(open(f, 'rb'))[1]
  dir = 'cp/cp_dataset/828.pkl'
  notes = pickle.load(open(dir, 'rb'))[1]
  print(notes)
  print(event2idx)
  for ev in notes:
    if ev[0] != 0:
      tmp = ev[0]
      print(event2idx.get(ev[0]))
    else:
      print('0')
    
  
