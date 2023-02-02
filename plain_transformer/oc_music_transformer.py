import torch
from torch import nn
import torch.nn.functional as F
from transformer_helpers import (
  generate_causal_mask,
  weights_init,
  TokenEmbedding, 
  PositionalEncoding
)

class PlainTransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, dropout=0.1, activation='relu'):
    super(PlainTransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation

    self.decoder_layers = nn.ModuleList()
    for i in range(n_layer):
      self.decoder_layers.append(
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
      )

  def forward(self, x):
    attn_mask = generate_causal_mask(x.size(0)).to(x.device)
    # print (attn_mask.size())
    out = x
    for i in range(self.n_layer):
      out = self.decoder_layers[i](out, src_mask=attn_mask)

    return out

class MusicTransformer(nn.Module):
  def __init__(self, n_token, n_layer, n_head, d_model, d_ff, d_embed,
    activation='relu', dropout=0.1, use_pe=True
  ):
    super(MusicTransformer, self).__init__()
    self.n_token = n_token
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation
    self.emb_sized = [160, 128, 32, 32, 416]
    self.pitch_token_emb = TokenEmbedding(self.n_token[0], self.emb_sized[0], self.emb_sized[0])
    self.vel_token_emb = TokenEmbedding(self.n_token[1], self.emb_sized[1], self.emb_sized[1])
    self.dur_token_emb = TokenEmbedding(self.n_token[2], self.emb_sized[2], self.emb_sized[2])
    self.pos_token_emb = TokenEmbedding(self.n_token[3], self.emb_sized[3], self.emb_sized[3])
    self.bar_token_emb = TokenEmbedding(self.n_token[4], self.emb_sized[4], self.emb_sized[4])
    self.d_embed = d_embed
    self.in_linear = nn.Linear(sum(self.emb_sized), self.d_embed)

    self.pe = PositionalEncoding(d_embed)
    self.pos_out_proj = nn.Linear(d_model, self.n_token[3])
    self.bar_out_proj = nn.Linear(d_model, self.n_token[4])
    self.pitch_out_proj = nn.Linear(d_model + self.emb_sized[3] + self.emb_sized[4], self.n_token[0])
    self.vel_out_proj = nn.Linear(d_model + self.emb_sized[3] + self.emb_sized[4], self.n_token[1])
    self.dur_out_proj = nn.Linear(d_model + self.emb_sized[3] + self.emb_sized[4], self.n_token[2])
    self.transformer_decoder = PlainTransformerDecoder(
      n_layer, n_head, d_model, d_ff, dropout, activation
    )

    self.emb_dropout = nn.Dropout(self.dropout)
    self.use_pe = use_pe
    self.apply(weights_init)

  def forward(self, x, y, keep_last_only=False, inference=False):
    x_pitch = self.pitch_token_emb(x[...,0])
    x_velocity = self.vel_token_emb(x[...,1])
    x_duration = self.dur_token_emb(x[...,2])
    x_position = self.pos_token_emb(x[...,3])
    x_bar = self.bar_token_emb(x[...,4])
    x_emb = torch.cat(
      [
        x_pitch,
        x_velocity,
        x_duration,
        x_position,
        x_bar,
      ], dim=-1)
    emb_linear = self.in_linear(x_emb)
    if self.use_pe:
      x_inp = self.emb_dropout(emb_linear) + self.pe(x.size(0))
    else:
      x_inp = self.emb_dropout(emb_linear)

    dec_out = self.transformer_decoder(x_inp)
    pos_dec_logits = self.pos_out_proj(dec_out)
    bar_dec_logits = self.bar_out_proj(dec_out)
    if not inference:
      y_position = self.pos_token_emb(y[...,3])
      y_bar = self.bar_token_emb(y[...,4])
    else:
      y_position = torch.argmax(pos_dec_logits, dim=-1, keepdim=True).squeeze(-1)
      y_bar = torch.argmax(bar_dec_logits, dim=-1, keepdim=True).squeeze(-1)
      y_position = self.pos_token_emb(y_position)
      y_bar = self.bar_token_emb(y_bar)
    #print(y_position.size(), y_bar.size(), dec_out.size())
    dec_out = torch.cat(
    [
      dec_out,
      y_position,
      y_bar,
    ], dim=-1)
    pitch_dec_logits = self.pitch_out_proj(dec_out)
    vel_dec_logits = self.vel_out_proj(dec_out)
    dur_dec_logits = self.dur_out_proj(dec_out)
    if keep_last_only:
      pos_dec_logits = pos_dec_logits[-1, :, :]
      bar_dec_logits = bar_dec_logits[-1, :, :]
      pitch_dec_logits = pitch_dec_logits[-1, :, :]
      vel_dec_logits = vel_dec_logits[-1, :, :]
      dur_dec_logits = dur_dec_logits[-1, :, :]
    print('--pitch shape : {}, --vel shape : {}, --dur shape : {}, --pos shape : {}, bar shape : {}'.format(pitch_dec_logits.size(), vel_dec_logits.size(), dur_dec_logits.size(), pos_dec_logits.size(), bar_dec_logits.size()))
    return pitch_dec_logits, vel_dec_logits, dur_dec_logits, pos_dec_logits, bar_dec_logits
# need modifying
  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    recons_loss = F.cross_entropy(
      dec_logits, dec_tgt, reduction=reduction).float()
    return recons_loss

