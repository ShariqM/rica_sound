import os
import subprocess
import atexit
import signal
import pdb
import sys

def get_model_num():
  f = open('last_run.log', 'r')
  run_num = int(f.readline()) + 1
  f.close()

  subprocess.Popen(
      ['cp hyperparameters.py hparams_logs/%d_hyperparameters.py' % run_num], shell=True)

  f = open('last_run.log', 'w')
  f.write(str(run_num))
  f.close()
  return run_num

def setup_tensorboard(logdir):
  print ("Logging to %s" % logdir) # Use tf.logging or something...

  # Start process, register inside a process group using setsid
  tb_process = subprocess.Popen(
      ["tensorboard --logdir=%s --port=54621" % (logdir)],
      shell=True, preexec_fn=os.setsid)
  def kill_tensorboard(*arg):
    print ("*** ||| *** Killing Tensorboard *** ||| ***")
    os.killpg(os.getpgid(tb_process.pid), signal.SIGTERM)  # Kill child if I die
    sys.exit(1)

  signal.signal(signal.SIGINT, kill_tensorboard)
  atexit.register(kill_tensorboard)
