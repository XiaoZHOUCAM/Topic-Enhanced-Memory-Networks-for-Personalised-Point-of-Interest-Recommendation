#encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""

import TLDA
import json

doc_num = 0
poi_num = 0
time_num = 0
topic_k = 7

input_data = []
doc_index = []
doc_len = []

f_i = open("sample_data")
readlines = f_i.readlines()
f_i.close()

for line in readlines:
    new_line = line.strip("\n").split("\t")
    ii = int(new_line[0])
    if ii > doc_num:
        doc_num = ii 
    json_data = json.loads(new_line[1])
    
    cur_data = []
    for item in json_data:
        cur_poi_index = item[0]
        cur_time_index = item[1]
        if cur_poi_index > poi_num:
            poi_num = cur_poi_index
        if cur_time_index > time_num:
            time_num = cur_time_index
        cur_item = (ii,item[0],item[1])
        cur_data.append(cur_item)
    doc_len.append(len(cur_data))
    doc_index.append(ii)
    input_data.append(cur_data)
doc_num = doc_num + 1
poi_num = poi_num + 1
time_num = time_num + 1

print("doc_num:",doc_num)
print("poi_num:",poi_num)
print("time_num:",time_num)
print(topic_k)

tlda = TLDA.tlda(input_data,doc_index,doc_len,doc_num,poi_num,time_num,topic_k,10)
tlda.train_model()
tlda.save_model_data("tlda_model_file")