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
    #[chord, tempo, barbeat, type, track, pitch, dur, vel]
    self.n_token = n_token #[134, 67, 18, 3, 4, 89, 17, 64]
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation
    self.emb_sized = [256, 128, 64, 32, 4, 512, 128, 256]
    self.chord_token_emb = TokenEmbedding(self.n_token[0], self.emb_sized[0], self.emb_sized[0])
    self.tempo_token_emb = TokenEmbedding(self.n_token[1], self.emb_sized[1], self.emb_sized[1])
    self.barbeat_token_emb = TokenEmbedding(self.n_token[2], self.emb_sized[2], self.emb_sized[2])
    self.type_token_emb = TokenEmbedding(self.n_token[3], self.emb_sized[3], self.emb_sized[3])
    self.track_token_emb = TokenEmbedding(self.n_token[4], self.emb_sized[4], self.emb_sized[4])
    self.pitch_token_emb = TokenEmbedding(self.n_token[5], self.emb_sized[5], self.emb_sized[5])
    self.dur_token_emb = TokenEmbedding(self.n_token[6], self.emb_sized[6], self.emb_sized[6])
    self.vel_token_emb = TokenEmbedding(self.n_token[7], self.emb_sized[7], self.emb_sized[7])
    self.d_embed = d_embed
    self.in_linear = nn.Linear(sum(self.emb_sized), self.d_embed)

    self.pe = PositionalEncoding(d_embed)
    self.chord_out_proj = nn.Linear(d_model, self.n_token[0])
    self.tempo_out_proj = nn.Linear(d_model, self.n_token[1])    
    self.barbeat_out_proj = nn.Linear(d_model, self.n_token[2])
    self.type_out_proj = nn.Linear(d_model, self.n_token[3])
    self.track_out_proj = nn.Linear(d_model, self.n_token[4])
    self.pitch_out_proj = nn.Linear(d_model, self.n_token[5])
    self.dur_out_proj = nn.Linear(d_model, self.n_token[6])
    self.vel_out_proj = nn.Linear(d_model, self.n_token[7])
    self.concat_proj = nn.Linear(self.d_model + 32, self.d_model)
    self.transformer_decoder = PlainTransformerDecoder(
      n_layer, n_head, d_model, d_ff, dropout, activation
    )

    self.emb_dropout = nn.Dropout(self.dropout)
    self.use_pe = use_pe
    self.apply(weights_init)

  def forward(self, x, y, keep_last_only=False, inference=False):
    x_chord = self.chord_token_emb(x[...,0])
    x_tempo = self.tempo_token_emb(x[...,1])
    x_barbeat = self.barbeat_token_emb(x[...,2])
    x_type = self.type_token_emb(x[...,3])
    x_track = self.track_token_emb(x[...,4])
    x_pitch = self.pitch_token_emb(x[...,5])
    x_dur = self.dur_token_emb(x[...,6])
    x_vel = self.vel_token_emb(x[...,7])
    #print(x_chord.size())
    #print(x_tempo.size())
    #print(x_barbeat.size())
    #print(x_type.size())
    #print(x_track.size())
    #print(x_pitch.size())
    #print(x_dur.size())
    #print(x_vel.size())
    x_emb = torch.cat(
      [
        x_chord,
        x_tempo,
        x_barbeat,
        x_type,
        x_track,
        x_pitch,
        x_dur,
        x_vel,
      ], dim=-1)
    #print(x_emb.size())
    emb_linear = self.in_linear(x_emb)
    if self.use_pe:
      x_inp = self.emb_dropout(emb_linear) + self.pe(x.size(0))
    else:
      x_inp = self.emb_dropout(emb_linear)

    dec_out = self.transformer_decoder(x_inp)
    y_type = self.type_out_proj(dec_out)
    if not inference:
      tgt_type = self.type_token_emb(y[...,2])
    else:
      tgt_type = torch.argmax(y_type, dim=-1, keepdim=True).squeeze(-1)
      tgt_type = self.type_token_emb(tgt_type)
    #print(y_position.size(), y_bar.size(), dec_out.size())
    y_concat = torch.cat([dec_out, tgt_type], dim=-1)
    y_ = self.concat_proj(y_concat)
    y_chord = self.chord_out_proj(y_)
    y_tempo = self.tempo_out_proj(y_)
    y_barbeat = self.barbeat_out_proj(y_)
    y_track = self.track_out_proj(y_)
    y_pitch = self.pitch_out_proj(y_)
    y_dur = self.dur_out_proj(y_)
    y_vel = self.vel_out_proj(y_)

    if keep_last_only:
      #shape : (s, b, f)
      y_chord = y_chord[-1, :, :]
      y_tempo = y_tempo[-1, :, :]
      y_barbeat = y_barbeat[-1, :, :]
      y_type = y_type[-1, :, :]
      y_track = y_track[-1, :, :]
      y_pitch = y_pitch[-1, :, :]
      y_dur = y_dur[-1, :, :]
      y_vel = y_vel[-1, :, :]
    #print('--chord shape : {}, --barbeat shape : {}, --type shape : {}, --pitch shape : {}, --duration shape : {}, --velocity shape'.format(
     #   y_chord.size(), y_barbeat.size(), y_type.size(), y_pitch.size(), y_dur.size(), y_vel.size()))
    return y_chord, y_tempo, y_barbeat, y_type, y_track, y_pitch, y_dur, y_vel
    
  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    recons_loss = F.cross_entropy(
      dec_logits, dec_tgt, reduction=reduction).float()
    return recons_loss

