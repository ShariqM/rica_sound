import numpy as np
import glob
import pdb
from scipy.io import wavfile

def get_wav_files(data_source):
  base = 'data/lewicki_audiodata'
  if data_source == "mix":
    wf1 = glob.glob('%s/envsounds/*.wav' % base)
    wf2 = glob.glob('%s/mammals/*.wav' % base)
    ratio = int(np.ceil(2*len(wf2)/len(wf1))) # 2 to 1 (env to mammals)
    return wf1 * ratio + wf2
  else:
    return glob.glob("%s/%s/*.wav" % (base, data_source))

def load_batch(hparams):
  """Load a batch of data of length input_size, normalize to [-1, 1]."""
  normalize_by = 2 ** 15  # Max value
  x_batch = np.zeros((hparams.batch_size, hparams.input_size))

  wav_files = get_wav_files(hparams.data_source)
  for i in range(hparams.batch_size):
    # Choose a random file
    wfile = np.random.choice(wav_files)
    Fs, x_raw = wavfile.read(wfile)

    # Choose a random input_size segment from the wav
    start = np.random.randint(len(x_raw) - hparams.input_size)
    x_batch[i, :] = x_raw[start : start + hparams.input_size]
    x_batch[i, :] /= normalize_by

  return x_batch
