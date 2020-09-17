import tensorflow as tf


class PolynomialDecayWithLinearWarmup(
        tf.keras.optimizers.schedules.PolynomialDecay):
    """
    Modified from https://github.com/tensorflow/addons/issues/2086
    """
    def __init__(self, target_learning_rate, warmup_steps,
                 decay_steps, **kwargs):
        super(PolynomialDecayWithLinearWarmup, self).__init__(
                initial_learning_rate=target_learning_rate,
                decay_steps=decay_steps,
                end_learning_rate=0.0, **kwargs)
        self.target_learning_rate = target_learning_rate
        self.warmup_steps = warmup_steps
        self.warmup_delta = self.target_learning_rate / self.warmup_steps

    def __call__(self, step):
        myname = "PolynomialDecayWithLinearWarmup"
        with tf.name_scope(self.name or myname):
            learning_rate = tf.cond(
                pred=tf.less(step, self.warmup_steps),
                true_fn=lambda:
                (self.warmup_delta * tf.cast(step, dtype=tf.float32)),
                false_fn=lambda: (super(PolynomialDecayWithLinearWarmup,
                                        self).__call__(
                                            step - self.warmup_steps)))
        return learning_rate

    def get_config(self):
        config = {
            "target_learning_rate": self.target_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        base_config = super(PolynomialDecayWithLinearWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
