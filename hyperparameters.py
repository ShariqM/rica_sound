class HyperParameters():
  # Model Parameters
  input_size = 128
  filter_size = input_size
  num_filters = input_size
  output_size = input_size

  # Data source
  # Options: 'mix', 'pitt, 'envsounds', 'mammals'
  data_source = 'mix'

  # Optimization Parameters
  batch_size = 512
  max_steps = 2 ** 16
  Lambda = 0.25  # XXX FIXME
  learning_rate = 2.0  # XXX FIXME
  bound_value = 1e5

  # Plotting
  plot_basis_frequency = 100
  save_frequency = 100

  norm_factor = 1
