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

    self.token_emb = TokenEmbedding(n_token, d_embed, d_model)
    self.d_embed = d_embed

    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(d_model, n_token)

    self.transformer_decoder = PlainTransformerDecoder(
      n_layer, n_head, d_model, d_ff, dropout, activation
    )

    self.emb_dropout = nn.Dropout(self.dropout)
    self.use_pe = use_pe
    self.apply(weights_init)

  def forward(self, x, keep_last_only=False):
    x_emb = self.token_emb(x)

    if self.use_pe:
      x_inp = self.emb_dropout(x_emb) + self.pe(x.size(0))
    else:
      x_inp = self.emb_dropout(x_emb)

    dec_out = self.transformer_decoder(x_inp)
    dec_logits = self.dec_out_proj(dec_out)

    if keep_last_only:
      dec_logits = dec_logits[-1, :, :]

    return dec_logits

  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction=reduction
    ).float()

    return {
      'recons_loss': recons_loss,
      'total_loss': recons_loss
    }
