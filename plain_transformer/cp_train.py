import sys, os, time, random
#from model.music_performer_sine_spe import MusicPerformerSineSPE
#from model.music_performer import MusicPerformer
from cp_music_transformer import MusicTransformer
from cp_dataloader_full_song import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader
from cp_validate import validate

from utils import pickle_load
from torch import nn, optim
import torch
import numpy as np

gpuid = 1
torch.cuda.set_device(gpuid)

train_steps = 0
max_steps = 400000
warmup_steps = 200
max_lr, min_lr = 1e-4, 5e-6

ckpt_dir = 'result_new/'
pretrained_param_path = ''
optimizer_path = ''
ckpt_interval = 10
val_interval = 1
log_interval = 100

def log_epoch(log_file, log_data, is_init=False):
  if is_init:
    with open(log_file, 'w') as f:
      f.write('{:4} {:8} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'ep_time'))

  with open(log_file, 'a') as f:
    f.write('{:<4} {:<8} {:<12} {:<12}\n'.format(
      log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['time'], 2)
    ))


def train_model(epoch, model, dloader, optim, sched):
  model.train()
  recons_loss_rec = 0.
  accum_samples = 0

  print ('[epoch {:03d}] training ...'.format(epoch))
  print ('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
  st = time.time()
  for batch_idx, batch_samples in enumerate(dloader):
    #print("now in batch {}".format(batch_idx))
    model.zero_grad()
    #print(batch_samples['dec_input'])
    # shape : (b, s, f) -> (s, b, f)
    batch_dec_inp = batch_samples['dec_input'].cuda(gpuid).permute(1, 0, 2)
    batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)

    batch_inp_lens = batch_samples['length']
    #print("loader complete!")
    # print ('[shapes]\n -- enc_inp: {}\n -- dec_inp: {}\n -- dec_tgt: {}\n -- bar_pos: {}\n -- length: {}\n -- pad_mask: {}'.format(
    #   batch_enc_inp.size(), batch_dec_inp.size(), batch_dec_tgt.size(), batch_inp_bar_pos.size(), batch_inp_lens, batch_padding_mask.size()
    # ))

    global train_steps
    train_steps += 1

    y_chord, y_tempo, y_barbeat, y_type, y_track, y_pitch, y_dur, y_vel = model(batch_dec_inp, batch_dec_tgt.permute(1, 0, 2))
    #print("after transformer")
    # print ('[output shapes] mu: {}, logvar: {}, logits: {}'.format(
    #   mu.size(), logvar.size(), dec_logits.size()
    # ))
    #shape(s, b, f) -> (b, f, s)
    y_chord = y_chord[:,...].permute(1, 2, 0)
    y_tempo = y_tempo[:,...].permute(1, 2, 0)
    y_barbeat = y_barbeat[:,...].permute(1, 2, 0)
    y_type = y_type[:,...].permute(1, 2, 0)
    y_track = y_track[:,...].permute(1, 2, 0)
    y_pitch = y_pitch[:,...].permute(1, 2, 0)
    y_dur = y_dur[:,...].permute(1, 2, 0)
    y_vel = y_vel[:,...].permute(1, 2, 0)
    print(y_chord.size(), y_tempo.size(), y_barbeat.size(), y_type.size(), y_track.size(), y_pitch.size(), y_dur.size(), y_vel.size())

    loss_chord = model.compute_loss(y_chord, batch_dec_tgt[...,0])
    loss_tempo = model.compute_loss(y_tempo, batch_dec_tgt[...,1])
    loss_barbeat = model.compute_loss(y_barbeat, batch_dec_tgt[...,2])
    loss_type = model.compute_loss(y_type, batch_dec_tgt[...,3])
    loss_track = model.compute_loss(y_track, batch_dec_tgt[...,4])
    loss_pitch = model.compute_loss(y_pitch, batch_dec_tgt[...,5])
    loss_dur = model.compute_loss(y_dur, batch_dec_tgt[...,6])
    loss_vel = model.compute_loss(y_vel, batch_dec_tgt[...,7])
    losses = {}
    losses['total_loss'] = (loss_chord + loss_tempo + loss_barbeat + loss_type + loss_track + loss_pitch + loss_dur + loss_vel ) / 8
    losses['recons_loss'] = (loss_chord + loss_tempo + loss_barbeat + loss_type + loss_track + loss_pitch + loss_dur + loss_vel ) / 8
    # clip gradient & update model
    losses['total_loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    recons_loss_rec += batch_samples['id'].size(0) * losses['recons_loss'].item()
    accum_samples += batch_samples['id'].size(0)

    # anneal learning rate
    if train_steps < warmup_steps:
      curr_lr = max_lr * train_steps / warmup_steps
      optim.param_groups[0]['lr'] = curr_lr
    else:
      sched.step(train_steps - warmup_steps)

    print (' -- epoch {:03d} | batch {:03d}: len: {}\n   * loss = {:.4f}, step = {}, time_elapsed = {:.2f} secs'.format(
      epoch, batch_idx, batch_inp_lens, recons_loss_rec / accum_samples, train_steps, time.time() - st
    ))
    print ('--chord loss {:.4f}, --tempo loss {:.4f}, --barbeat loss {:.4f}, --type loss {:.4f}, --track loss {:.4f}, --pitch loss {:.4f}, --dur loss {:.4f}, --vel loss {:.4f}'.format(
        loss_chord, loss_tempo, loss_barbeat, loss_type, loss_track, loss_pitch, loss_dur, loss_vel))

    if not train_steps % log_interval:
      log_data = {
        'ep': epoch,
        'steps': train_steps,
        'recons_loss': recons_loss_rec / accum_samples,
        'time': time.time() - st
      }
      log_epoch(
        os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
      )

  print ('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
    epoch, recons_loss_rec / accum_samples, time.time() - st
  ))
  log_data = {
    'ep': epoch,
    'steps': train_steps,
    'recons_loss': recons_loss_rec / accum_samples,
    'time': time.time() - st
  }
  log_epoch(
    os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
  )

  return recons_loss_rec / accum_samples

if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    './cp_dataset', 'pop909_cp_Chord_vocab.pkl', 
    do_augment=True, model_enc_seqlen=128, model_dec_seqlen=512, model_max_bars=16,
    pieces=pickle_load('./splits/train_pieces.pkl'),
    pad_to_same=True
  )
  val_dset = REMIFullSongTransformerDataset(
    './cp_dataset', 'pop909_cp_Chord_vocab.pkl', 
    do_augment=False, model_enc_seqlen=128, model_dec_seqlen=512, model_max_bars=16,
    pieces=pickle_load('./splits/val_pieces.pkl'),
    pad_to_same=True
  )
  print (len(dset.pieces))
  dloader = DataLoader(dset, batch_size=8, shuffle=True, num_workers=16)
  val_dloader = DataLoader(val_dset, batch_size=8, shuffle=True, num_workers=16)
  n_token = [134, 67, 18, 3, 4, 89, 17, 64]
  model = MusicTransformer(
    n_token, 12, 8, 256, 1024, 256
  ).cuda(gpuid)

  if pretrained_param_path:
    pretrained_dict = torch.load(pretrained_param_path)
    pretrained_dict = {
      k:v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
    }
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

  model.train()
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print ('# params:', n_params)

  opt_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adam(opt_params, lr=max_lr)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, max_steps, eta_min=min_lr
  )
  if optimizer_path:
    optimizer.load_state_dict(
      torch.load(optimizer_path)
    )

  params_dir = os.path.join(ckpt_dir, 'params/')
  optimizer_dir = os.path.join(ckpt_dir, 'optim/')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  if not os.path.exists(params_dir):
    os.makedirs(params_dir)
  
  if not os.path.exists(optimizer_dir):
    os.makedirs(optimizer_dir)

  for ep in range(2000):
    loss = train_model(ep+1, model, dloader, optimizer, scheduler)
    if not (ep + 1) % ckpt_interval:
      torch.save(model.state_dict(),
        os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep+1, loss))
      )
      torch.save(optimizer.state_dict(),
        os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep+1, loss))
      )
    if not (ep + 1) % val_interval:
      val_losses = validate(model, val_dloader, rounds=8, gpuid=gpuid)
      with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
        f.write("ep{:03d} | loss: {:.3f} valloss: {:.3f} (+/- {:.3f})\n".format(
          ep + 1, loss, np.mean(val_losses), np.std(val_losses)
        ))

