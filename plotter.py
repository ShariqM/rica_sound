import matplotlib as mpl
mpl.use('Agg')  # No display
import matplotlib.pyplot as plt

import numpy as np
import io
import pdb

def get_plot_buf(filters):

  ## XXX Plot a subset for now
  num_rows = 6
  num_cols = 6
  #num_rows = int(filters.shape[0] / 8)
  #num_cols = int(filters.shape[0] / num_rows)
  figsize = (num_cols * 2, num_rows)
  figure, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

  for i in range(num_rows):
    for j in range(num_cols):
      idx = i * num_cols + j
      axes[i, j].plot(filters[idx])
      axes[i, j].xaxis.set_visible(False)
      axes[i, j].yaxis.set_visible(False)
  plt.tight_layout()

  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close()

  buf.seek(0)
  return buf.getvalue()

