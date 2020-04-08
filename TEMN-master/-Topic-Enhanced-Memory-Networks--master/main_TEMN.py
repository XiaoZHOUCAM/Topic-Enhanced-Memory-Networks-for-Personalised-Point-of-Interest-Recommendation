#encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""

import TEMN
import random
import tensorflow as tf
import process_TEMN_data

all_data, user_num, poi_num, topic_k, all_test_data = process_TEMN_data.process_fun("sample_data")
random.shuffle(all_data)

num_users = user_num
num_items = poi_num
topic_num = topic_k

class args:
    std = 0.1
    num_mem = 10
    embedding_size = 50
    constraint = True
    rnn_type = set(['PAIR'])
    margin = 0.1
    topic_num = topic_num
    l2_reg = 0.00001
    opt = 'SGD'
    clip_norm = 2
    dropout = 0.7
    learn_rate = 0.01
    max_p_num = 100
    stddev = 0.1
    lamb_m = 0.1
    lamb_d = 0.1
    ratio1 = 0.1
    ratio2 = 0.1
    init_method = "normal"


ar = args()
user_num = num_users
item_num = num_items

model = TEMN.TEMN(user_num,item_num,ar)
saver = tf.train.get_or_create_global_step()
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

print("build finish")

n_sample = len(all_data)
n_sample_test = len(all_test_data)
batch_size = 64
batch_num = int((n_sample + batch_size - 1)/batch_size)
batch_num_test = int((n_sample_test + batch_size - 1)/batch_size)

print("batch_num", batch_num)
print("batch_num_test", batch_num_test)

Iter = 10

for it in range(Iter):
    for i in range(batch_num):
        beg = i*batch_size
        end = min((i + 1)*batch_size, n_sample)
        cur_train_data = all_data[beg:end]
        feed_dict = model.get_list_feed_dict(cur_train_data)
        cost = sess.run([model.cost, model.dist_cost, model.mem_cost, model.topic_cost, model.train_op], feed_dict)
        if i % 50 == 0:
            print(it, i, cost[0])


# Test
print("start testing")
def get_topk(scores):
    f = scores[0]
    sort_scores = sorted(scores, reverse=True)
    cur_index = sort_scores.index(f)
    return cur_index


f_out = open("top_k_label", "w")
f_o = open("test_scores", "w")
ii_x = 0

for t_data in all_test_data:
    cur_test_data = t_data
    feed_dict_test = model.get_list_feed_dict(cur_test_data, "")
    scores = sess.run(model.predict_op, feed_dict_test)
    cur_pos = get_topk(scores)
    f_out.write(str(cur_pos) + "\n")
    for j in range(len(cur_test_data)):
        cur_u = cur_test_data[j][0]
        cur_i = cur_test_data[j][1]
        cur_s = scores[j]
        r_line = str(cur_u) + "\t" + str(cur_i) + "\t" + str(cur_s)
        f_o.write(r_line + "\n")
    ii_x += 1
    if ii_x % 10 == 0:
        print(ii_x)

f_o.close()
f_out.close()