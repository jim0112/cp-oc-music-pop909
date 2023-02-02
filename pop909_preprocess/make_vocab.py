import pickle
from chorder import Chord
if __name__ == "__main__":
  # vocab = pickle.load(open('ls_orig_vocab.pkl', 'rb')).event2idx
  # print (vocab)
  new_vocab = set()
  
  # for k, v in sorted(vocab.items(), key=lambda x: x[1]):
  #   print ('{:03d} : {}'.format(v, k))
  for pitch in range(21, 109):
    new_vocab.add('Note_Pitch_{}'.format(pitch))
  for duration in range(40, 1921, 40):
    new_vocab.add('Note_Duration_{}'.format(duration))
  for beat in range(16):
    new_vocab.add('Beat_16_{}'.format(beat))
  for beat in range(12):
    new_vocab.add('Beat_12_{}'.format(beat))
  for tempo in range(32, 227, 3):
    new_vocab.add('Tempo_{}'.format(tempo))
  for velocity in range(1, 128):
    new_vocab.add('Note_Velocity_{}'.format(velocity))
  for chord in Chord.standard_qualities:
    for i in range(0, 12):
      new_vocab.add('Chord_{}_{}'.format(i, chord))
  new_vocab.add('Bar_None')
  new_vocab.add('EOS_None')
  new_vocab.add('Chord_N_N')
  events = sorted(list(new_vocab))
  event2idx, idx2event = dict(), dict()
  for i, ev in enumerate(events):
    event2idx[ ev ] = i
    idx2event[ i ] = ev
    print('{:03d} : {}'.format(i, idx2event[i]))

  pickle.dump(
    tuple((event2idx, idx2event)), 
    open('pop909_triplet_vocab.pkl', 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL
  )