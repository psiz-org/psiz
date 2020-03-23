# Compare current loss to best loss.
                if loss_val < loss_val_best:
                    last_improvement_stop = tf.constant(0, dtype=tf.int32)
                else:
                    last_improvement_stop = (
                        last_improvement_stop + tf.constant(1, dtype=tf.int32)
                    )

                if loss_val < loss_val_best:
                    last_improvement_reduce = tf.constant(0, dtype=tf.int32)
                else:
                    last_improvement_reduce = (
                        last_improvement_reduce +
                        tf.constant(1, dtype=tf.int32)
                    )

                if loss_val < loss_val_best:
                    loss_train_best = loss_train
                    loss_val_best = loss_val
                    epoch_best = epoch + tf.constant(1, dtype=tf.int64)
                    z_best = model.layers[4].z
                    attention_best = model.layers[4].attention
                    theta_best = model.layers[4].theta

                if last_improvement_stop >= tf_patience:
                    break
    

    patience=10, n_max_epoch=5000
    tf_max_epoch = tf.constant(max_epoch, dtype=tf.int64)  # TODO move out
    tf_patience = tf.constant(patience, dtype=tf.int32)  # TODO move out

    last_improvement_stop = tf.constant(0, dtype=tf.int32)
    last_improvement_reduce = tf.constant(0, dtype=tf.int32)


    # best_weights = model.get_weights()
    # model.set_weights(best_weights)

    # Initialize best values. TODO
    # epoch_best = tf.constant(0, dtype=tf.int64)
    # loss_train_best = tf.constant(np.inf, dtype=K.floatx())
    # loss_val_best = tf.constant(np.inf, dtype=K.floatx())

    class EarlyStopping(Callback):
  """Stop training when a monitored metric has stopped improving.
  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be 'loss', and mode would be 'min'. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.
  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.
  Example:
  >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the validation loss for three consecutive epochs.
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, batch_size=1, callbacks=[callback],
  ...                     verbose=0)
  >>> len(history.history['loss'])  # Only 4 epochs are run.
  4
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False):
    """Initialize an EarlyStopping callback.
    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning('Early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    return monitor_value


