import sys, os, time, random
sys.path.append('./model')

# from model.vae_fast_transformer import MusicFastOptimus
#from model.music_performer import MusicPerformer
from cp_dataloader_full_song import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader

from utils import device, pickle_load
from torch import nn, optim
import torch
import numpy as np

gpuid = 0
torch.cuda.set_device(gpuid)
redraw_pct = 0.05

#pretrained_dir = 'ckpt_fast_plain/1229_l24_performer_random_redraw'
#pretrained_paths = [
#  os.path.join(pretrained_dir, 'params/{}'.format(x)) for x in os.listdir(pretrained_dir + '/params')\
#  if '.pt' in x
#]

def validate(model, dloader, rounds=4, gpuid=gpuid):
  model.eval()
  loss_rec = []

  with torch.no_grad():
    for r in range(rounds):
      print (' >> validating ... (round {})'.format(r+1))
      for batch_idx, batch_samples in enumerate(dloader):
        batch_dec_inp = batch_samples['dec_input'].cuda(gpuid).permute(1, 0, 2)
        batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
        batch_inp_lens = batch_samples['length']

        # print ('[decoder input bounds]', torch.min(batch_dec_tgt), torch.max(batch_dec_tgt))
        print ('[shapes]\n -- dec_inp: {}\n -- dec_tgt: {}\n -- length: {}'.format(
          batch_dec_inp.size(), batch_dec_tgt.size(), batch_inp_lens
        ))

        y_chord, y_tempo, y_barbeat, y_type, y_track, y_pitch, y_dur, y_vel = model(batch_dec_inp, batch_dec_tgt.permute(1, 0, 2))
        #shape(s, b, f) -> (b, f, s)
        y_chord = y_chord[:,...].permute(1, 2, 0)
        y_tempo = y_tempo[:,...].permute(1, 2, 0)
        y_barbeat = y_barbeat[:,...].permute(1, 2, 0)
        y_type = y_type[:,...].permute(1, 2, 0)
        y_track = y_track[:,...].permute(1, 2, 0)
        y_pitch = y_pitch[:,...].permute(1, 2, 0)
        y_dur = y_dur[:,...].permute(1, 2, 0)
        y_vel = y_vel[:,...].permute(1, 2, 0)
        #print(y_chord.size(), y_tempo.size(), y_barbeat.size(), y_type.size(), y_track.size(), y_pitch.size(), y_dur.size(), y_vel.size())

        loss_chord = model.compute_loss(y_chord, batch_dec_tgt[...,0])
        loss_tempo = model.compute_loss(y_tempo, batch_dec_tgt[...,1])
        loss_barbeat = model.compute_loss(y_barbeat, batch_dec_tgt[...,2])
        loss_type = model.compute_loss(y_type, batch_dec_tgt[...,3])
        loss_track = model.compute_loss(y_track, batch_dec_tgt[...,4])
        loss_pitch = model.compute_loss(y_pitch, batch_dec_tgt[...,5])
        loss_dur = model.compute_loss(y_dur, batch_dec_tgt[...,6])
        loss_vel = model.compute_loss(y_vel, batch_dec_tgt[...,7])
        loss = (loss_chord + loss_tempo + loss_barbeat + loss_type + loss_track + loss_pitch + loss_dur + loss_vel ) / 8
        # print ('[output shapes] mu: {}, logvar: {}, logits: {}'.format(
        #   mu.size(), logvar.size(), dec_logits.size()
        # ))
        print('[chord loss] : {:.4f}, [tempo loss] : {:.4f}, [barbeat loss] : {:.4f}, [type loss] : {:.4f}, [track loss] : {:.4f}, [pitch loss] : {:.4f}, [duration loss] : {:.4f}, [velocity loss] : {:.4f}'.format(
          loss_chord, loss_tempo, loss_barbeat, loss_type, loss_track, loss_pitch, loss_dur, loss_vel))
        # print ('batch #{}:'.format(batch_idx + 1), round(losses['recons_loss'].item(), 3))
        loss_rec.append(loss.item())
    
  return loss_rec

if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    './remi_dataset', './pickles/remi_wenyi_vocab.pkl', 
    do_augment=False, model_enc_seqlen=128, model_dec_seqlen=10240, model_max_bars=512,
    pieces=pickle_load('./splits/val_pieces.pkl')[:100],
    pad_to_same=True
  )
  print (len(dset.pieces))
  dloader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=24)

  for cp in pretrained_paths:
    print ('>> now validating: {}'.format(cp))
    model = MusicPerformer(
      dset.vocab_size, 24, 8, 512, 2048, 512
    ).cuda(gpuid)
    pretrained_dict = torch.load(cp)
    pretrained_dict = {
      k:v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
    }
    # for k, v in pretrained_dict.items():
    #   print ('[key]', k)
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ('# params:', n_params)

    losses = validate(model, dloader)
    print ('valloss = {:.3f}, std = {:.3f}'.format(np.mean(losses), np.std(losses)))

    with open(os.path.join(pretrained_dir, 'valloss.txt'), 'a') as f:
      f.write("ckpt: {} | valloss: {:.3f} (+/- {:.3f})\n".format(
        cp.split('/')[-1], np.mean(losses), np.std(losses)
      ))
