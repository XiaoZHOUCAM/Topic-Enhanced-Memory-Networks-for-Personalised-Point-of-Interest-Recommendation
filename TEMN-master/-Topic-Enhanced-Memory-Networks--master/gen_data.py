#encoding:utf-8
'''
xz331@cam.ac.uk
abnerzxzhao@tencent.com
'''

import json
import random

doc_num = 5000
word_num = 10
time_num = 7
train_k = 100
test_k = 3
neg_k = 10

f_o = open("sample_data","w")
f_poi = open("poi2_xy","w")

for i in range(doc_num):
    train_data = []
    test_data = []
    neg_data = []
    for j in range(train_k):
        pos_id = abs(int(random.gauss(3,10)))%word_num
        time_id = abs(int(random.gauss(3,3)))%time_num
        neg_id = abs(int(random.gauss(7,10)))%word_num
        cur_item = [pos_id,time_id,neg_id]
        train_data.append([pos_id,time_id,neg_id])
    for k in range(test_k):
        pos_id = abs(int(random.gauss(3,10)))%word_num
        time_id = abs(int(random.gauss(3,3)))%time_num
        neg_id = abs(int(random.gauss(7,10)))%word_num
        cur_item = [pos_id,time_id,neg_id]
        test_data.append([pos_id,time_id,neg_id])
    for n in range(neg_k):
        neg_id = abs(int(random.gauss(7,10)))%word_num
        neg_data.append(neg_id)
    r_line = str(i) + "\t" + json.dumps(train_data) + "\t" + json.dumps(test_data) + "\t" + json.dumps(neg_data) + "\n"
    f_o.write(r_line)
f_o.close()

for poi in range(word_num):
    cur_x = random.uniform(115,117)
    cur_y = random.uniform(39,41)
    r_line = str(poi) + "\t" +  str(cur_x) +  "\t" + str(cur_y) + "\n"
    f_poi.write(r_line)
f_poi.close()