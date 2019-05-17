#encoding:utf-8
import tensorflow as tf

class basemodel(object):
    def __init__(self, args):
        self.args = args

    def _get_initializer(self):
        if self.args.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=self.args.stddev)
        elif self.args.init_method == 'uniform':
            return tf.random_uniform_initializer(-self.args.stddev, self.args.stddev)
        elif self.args.init_method == 'normal':
            return tf.random_normal_initializer(stddev=self.args.stddev)
        elif self.args.init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif self.args.init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif self.args.init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)
        elif self.args.init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=self.args.stddev)

    def _set_opt(self):
        if (self.args.opt == 'SGD'):
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
        elif (self.args.opt == 'Adam'):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        elif (self.args.opt == 'Adadelta'):
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
        elif (self.args.opt == 'Adagrad'):
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.learn_rate,
                                                 initial_accumulator_value=0.9)
        elif (self.args.opt == 'RMS'):
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate,
                                                 decay=0.9, epsilon=1e-6)
        elif (self.args.opt == 'Moment'):
            self.opt = tf.train.MomentumOptimizer(self.args.learn_rate, 0.9)



