import tensorflow as tf
import loader

def get_learning_rate(hparams, step):
  """Turn down the learning rate later."""
  return hparams.learning_rate

def should_plot_basis(hparams, step):
  return step and step % hparams.plot_basis_frequency == 0

def setup(hparams, sess, init_op, saver):
  sess.run(init_op)
  writer = tf.summary.FileWriter(hparams.logdir + "/train", sess.graph)
  if hparams.load_model_num:
    print ("Loading: %s" % hparams.loaddir)
    saver.restore(sess, tf.train.latest_checkpoint(hparams.loaddir))
  return writer

def run_train(hparams, x, x_target, basis_buffer, loss, basis_summaries, summaries):
  """Train the auto encoder RICA model."""
  learning_rate = tf.placeholder(tf.float32, shape=[])
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train = optimizer.minimize(loss)

  synthesis = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=hparams.synthesis_name)[0]
  norm_s_op = synthesis.assign(tf.nn.l2_normalize(synthesis, dim=1))

  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    writer = setup(hparams, sess, init_op, saver)

    for step in range(hparams.max_steps):
      x_batch = loader.load_batch(hparams)
      feed_dict = {x: x_batch, x_target: x_batch,
                   learning_rate: get_learning_rate(hparams, step)}

      if not should_plot_basis(hparams, step):
        result = sess.run([loss, train] + summaries, feed_dict=feed_dict)
      else:
        print ("Plotting basis functions")
        result = sess.run([loss, train] + basis_summaries + summaries,
                          feed_dict=feed_dict)

      (raw_loss, raw_summaries) = result[0], result[2:]
      for raw_summary in raw_summaries:
        writer.add_summary(raw_summary, step)
      print ("%d) Loss: %.3f" % (step, raw_loss))

      # Normalize synthesis weights to prevent sparsity degeneracy
      sess.run(norm_s_op)

      if step and step % hparams.save_frequency == 0:
        saver.save(sess, hparams.savedir, step)
        print ("Model Saved.")
      writer.flush()
