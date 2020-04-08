# encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
from __future__ import division
import tensorflow as tf
from basemodel import basemodel


class TEMN(basemodel):
    """
    @num_users:number of users
    @num_items:number of POIs
    @lamb_m:the margin of memory network
    @lamb_d:the margin of dis_model
    @ratio1:the influence of topic_model
    @ratio2:the influence of dis_model
    """

    def __init__(self, num_users, num_items, args):
        print('creating my TEMN!')
        self.num_users = num_users
        self.num_items = num_items
        self.graph = tf.Graph()
        self.args = args
        self.stddev = self.args.stddev
        self.learn_rate = self.args.learn_rate
        self.lamb_m = args.lamb_m
        self.lamb_d = args.lamb_d
        self.ratio1 = args.ratio1
        self.ratio2 = args.ratio2
        self.attention = None
        self.selected_memory = None
        self.num_mem = self.args.num_mem

        self.initializer = self._get_initializer()
        self._set_opt()
        self._creat_model_inputs()
        self._build_list_network()

    def get_list_feed_dict(self, batch, mode='training'):
        def process_all_items(x_all_item):
            cur_all_ii = [0 for j in range(self.args.max_p_num)]
            for j in range(len(x_all_item)):
                if j < self.args.max_p_num:
                    cur_all_ii[j] = x_all_item[j]
            return cur_all_ii

        if mode == 'training':
            user_input = [x[0] for x in batch]
            item_input = [x[1] for x in batch]
            uindist = [x[2] for x in batch]
            all_items_data = [process_all_items(x[3]) for x in batch]
            ll = [len(x[3]) for x in batch]
            item_input_neg = [x[4] for x in batch]
            topic_input = [x[5] for x in batch]
            uindistneg = [x[6] for x in batch]
            feed_dict = {
                self.user_input: user_input,  # user id
                self.item_input: item_input,  # item id
                self.item_input_neg: item_input_neg,
                self.L: ll,  # number of POIs a user visited
                self.all_items: all_items_data,  # POI list a user has visited
                self.label: topic_input,  # user-topic from TLDA
                self.DIST: uindist,  # distance from user to positive POI
                self.DIST_neg: uindistneg,  # distance from user to negative POI
                self.dropout: self.args.dropout
            }
        else:
            user_input = [x[0] for x in batch]
            item_input = [x[1] for x in batch]
            uindist = [x[2] for x in batch]
            all_items_data = [process_all_items(x[3]) for x in batch]
            ll = [len(x[3]) for x in batch]
            feed_dict = {
                self.user_input: user_input,
                self.item_input: item_input,
                self.L: ll,
                self.DIST: uindist,
                self.all_items: all_items_data,
                self.dropout: 1
            }
        feed_dict[self.learn_rate] = self.args.learn_rate
        return feed_dict

    def _creat_model_inputs(self):
        self.user_input = tf.placeholder(tf.int32, shape=[None], name='user')
        self.item_input = tf.placeholder(tf.int32, shape=[None], name='item')
        self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name='item_neg')
        self.input_type = tf.placeholder(tf.int32, shape=[None], name='type')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.label = tf.placeholder(tf.float32, shape=[None, self.args.topic_num], name='labels')

        self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        self.L = tf.placeholder(tf.float32, shape=[None], name='L')
        self.DIST = tf.placeholder(tf.float32, shape=[None], name='DIST')
        self.DIST_neg = tf.placeholder(tf.float32, shape=[None], name='DIST_neg')
        self.all_items = tf.placeholder(tf.int32, shape=[None, self.args.max_p_num], name="HISTORY")
        self.batch_size = tf.shape(self.item_input)[0]

    def _composition_layer(self, user_emb, item_emb, dist='L2', selected_memory=None):
        energy = item_emb - (user_emb + selected_memory)
        if 'L2' in dist:
            final_layer = -tf.sqrt(tf.reduce_sum(tf.square(energy), 1) + 1E-3)
        elif 'L1' in dist:
            final_layer = -tf.reduce_sum(tf.abs(energy), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer = tf.reshape(final_layer, [-1, 1])
        return final_layer

    def _get_prediction(self, user_emb, item_emb, memory_key):
        _key = tf.multiply(self.user_emb, self.item_emb)
        _key = tf.expand_dims(_key, 1)
        key_attention = tf.squeeze(tf.matmul(_key, memory_key))
        key_attention = tf.nn.softmax(key_attention)
        selected_memory = tf.matmul(key_attention, self.memory_value)
        final_layer = self._composition_layer(user_emb, item_emb, selected_memory=selected_memory)
        return final_layer

    def _build_list_network(self):
        stddev = self.stddev
        with tf.variable_scope('embedding_layer', initializer=self.initializer):
            with tf.device('/cpu:0'):
                self.user_item_key = tf.Variable(
                    tf.random_normal(
                        [self.args.embedding_size, self.num_mem],
                        stddev=stddev))
                self.memories = tf.Variable(
                    tf.random_normal(
                        [self.num_mem, self.args.embedding_size],
                        stddev=stddev))

                self.item_embeddings = tf.get_variable('item_emb', [self.num_items + 1, self.args.embedding_size],
                                                       initializer=self.initializer)
                self.all_items_emb = tf.nn.embedding_lookup(self.item_embeddings, self.all_items)
                self.item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
                self.item_emb_neg = tf.nn.embedding_lookup(self.item_embeddings, self.item_input_neg)
                self.dis_W = tf.get_variable("W", [self.num_users + 1, 1], initializer=self.initializer)
                self.dis_b = tf.get_variable("b", [self.num_users + 1, 1], initializer=self.initializer)
                self.dis_W_item = tf.get_variable("W_item", [self.num_items + 1, 1], initializer=self.initializer)

                if self.args.constraint:
                    self.all_items_emb = tf.clip_by_norm(self.all_items_emb, 1.0, axes=1)
                    self.item_emb = tf.clip_by_norm(self.item_emb, 1.0, axes=1)
                    self.item_emb_neg = tf.clip_by_norm(self.item_emb_neg, 1.0, axes=1)

                self.all_items_emb = tf.transpose(self.all_items_emb, perm=[0, 2, 1])
                self.cur_mask = tf.sequence_mask(self.L, self.args.max_p_num)
                self.cur_mask = tf.expand_dims(self.cur_mask, -1)
                self.cur_mask = tf.transpose(self.cur_mask, perm=[0, 2, 1])
                kept_indices = tf.cast(self.cur_mask, dtype=tf.float32)
                self.all_items_emb = self.all_items_emb * kept_indices
                self.user_emb_sum = tf.reduce_sum(self.all_items_emb, 2)
                self.LL = tf.expand_dims(self.L, -1)
                self.user_emb = self.user_emb_sum / self.LL

                self.item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)

                self.user_topic_W = tf.Variable(
                    tf.random_normal([self.args.embedding_size, self.args.topic_num], stddev=stddev))
                self.user_topic_b = tf.Variable(tf.random_normal([self.args.topic_num], stddev=stddev))
                self.topic_out = tf.matmul(self.user_emb, self.user_topic_W) + self.user_topic_b
                self.predict_topic = tf.nn.softmax(self.topic_out)
                self.topic_cost = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.topic_out, labels=self.label))

                self._key = tf.multiply(self.user_emb, self.item_emb)
                self.key_attention = tf.matmul(self._key, self.user_item_key)
                self.key_attention = tf.nn.softmax(self.key_attention)
                self.selected_memory = tf.matmul(self.key_attention, self.memories)
                final_layer = self._composition_layer(self.user_emb, self.item_emb,
                                                      selected_memory=self.selected_memory)
                final_layer_neg = self._composition_layer(self.user_emb, self.item_emb_neg,
                                                          selected_memory=self.selected_memory)
                self.predict_op = tf.squeeze(final_layer)
                self.mem_cost = tf.reduce_sum(tf.nn.relu((tf.squeeze(final_layer_neg - final_layer) + self.lamb_m)))

                self.dis_W_emb = tf.squeeze(tf.nn.embedding_lookup(self.dis_W, self.user_input))
                self.dis_b_emb = tf.squeeze(tf.nn.embedding_lookup(self.dis_b, self.user_input))
                self.dist_W_item_emb = tf.squeeze(tf.nn.embedding_lookup(self.dis_W_item, self.item_input))
                self.dist_W_item_emb_neg = tf.squeeze(tf.nn.embedding_lookup(self.dis_W_item, self.item_input_neg))
                self.Wis = (self.dis_W_emb * self.DIST + self.dis_b_emb + self.dist_W_item_emb * self.DIST)
                self.Wis_neg = (
                            self.dis_W_emb * self.DIST_neg + self.dis_b_emb + self.dist_W_item_emb_neg * self.DIST_neg)
                self.dist_cost = tf.reduce_sum(tf.nn.relu((self.lamb_d - self.Wis + self.Wis_neg)))

                self.cost = self.mem_cost + self.topic_cost * self.ratio1 + self.dist_cost * self.ratio2
                if self.args.l2_reg > 0:
                    vars = tf.trainable_variables()
                    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.args.l2_reg
                    self.cost += lossL2
                if self.args.opt == 'SGD':
                    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
                elif self.args.opt == 'Adam':
                    self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
                elif self.args.opt == 'Adadelta':
                    self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
                elif self.args.opt == 'Adagrad':
                    self.opt = tf.train.AdagradOptimizer(learning_rate=self.learn_rate, initial_accumulator_value=0.9)
                elif self.args.opt == 'RMS':
                    self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate, decay=0.9, epsilon=1e-6)
                elif self.args.opt == 'Moment':
                    self.opt = tf.train.MomentumOptimizer(self.args.learn_rate, 0.9)
                # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)
                gradients = self.opt.compute_gradients(self.cost)
                self.gradients = gradients

                def ClipIfNotNone(grad):
                    if grad is None:
                        return grad
                    grad = tf.clip_by_value(grad, -10, 10, name=None)
                    return tf.clip_by_norm(grad, self.args.clip_norm)

                if self.args.clip_norm > 0:
                    clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                else:
                    clipped_gradients = [(grad, var) for grad, var in gradients]

                # grads, _ = tf.clip_by_value(tf.gradients(self.cost, tvars),-10,10)
                self.optimizer = self.opt.apply_gradients(clipped_gradients)
                self.train_op = self.optimizer
