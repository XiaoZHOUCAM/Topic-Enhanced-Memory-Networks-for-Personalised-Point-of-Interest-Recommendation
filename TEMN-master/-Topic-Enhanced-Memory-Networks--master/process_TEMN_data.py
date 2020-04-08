# encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
import json
import math
import pickle


def get_dis(node1, node2):
    x = node1[0] - node2[0]
    y = node1[1] - node2[1]
    return math.sqrt(x * x + y * y)


def get_poi_xy():
    poi2xy = {}
    f_i = open("poi2_xy")
    readlines = f_i.readlines()
    f_i.close()
    for line in readlines:
        new_line = line.strip().split("\t")
        poi = int(new_line[0])
        x = float(new_line[1])
        y = float(new_line[2])
        poi2xy[poi] = [x, y]
    return poi2xy


def process_fun(file_name):
    poi2xy = get_poi_xy()

    document_topic, time_topic, topic_word = pickle.load(open("tlda_model_file", "rb"))
    print("document_topic,", document_topic.shape)
    topic_k = document_topic.shape[1]
    f_i = open(file_name)
    readlines = f_i.readlines()
    f_i.close()
    ret = []
    user_num = 0
    poi_num = 0
    ret_test = []
    for line in readlines:
        new_line = line.strip().split("\t")
        cur_doc_id = int(new_line[0])
        cur_data = json.loads(new_line[1])
        test_data = json.loads(new_line[2])
        neg_datas = json.loads(new_line[3])
        all_i = set()
        cur_c = 0
        poi_x = 0
        poi_y = 0
        for item in cur_data:
            if item[0] not in poi2xy:
                continue
            all_i.add(item[0])
            poi_x += poi2xy[item[0]][0]
            poi_y += poi2xy[item[0]][1]
            cur_c += 1
        if cur_c < 1:
            continue
        poi_x = poi_x / cur_c
        poi_y = poi_y / cur_c
        if cur_doc_id > user_num:
            user_num = cur_doc_id
        for item in cur_data:
            if item[0] > poi_num:
                poi_num = item[0]
            if item[2] > poi_num:
                poi_num = item[2]
            if item[0] not in poi2xy or item[2] not in poi2xy:
                continue
            cur_u = cur_doc_id
            cur_i = item[0]
            cur_x = poi2xy[cur_i][0]
            cur_y = poi2xy[cur_i][1]
            neg_i = item[2]
            neg_x = poi2xy[neg_i][0]
            neg_y = poi2xy[neg_i][1]
            opt_dis = get_dis([poi_x, poi_y], [cur_x, cur_y])
            neg_dis = get_dis([poi_x, poi_y], [neg_x, neg_y])
            cur_topic = document_topic[cur_u]
            cur_all_item = []
            for p in all_i:
                if p == cur_i:
                    continue
                cur_all_item.append(p)
            if len(cur_all_item) > 0:
                xx = cur_u, cur_i, opt_dis, cur_all_item, neg_i, cur_topic, neg_dis
                ret.append(xx)

        for item in test_data:
            cur_test_data = []
            if item[0] > poi_num:
                poi_num = item[0]
            if item[2] > poi_num:
                poi_num = item[2]
            if item[0] not in poi2xy or item[2] not in poi2xy:
                continue
            cur_u = cur_doc_id
            cur_i = item[0]
            cur_x = poi2xy[cur_i][0]
            cur_y = poi2xy[cur_i][1]
            opt_dis = get_dis([poi_x, poi_y], [cur_x, cur_y])
            cur_all_item = []
            for p in all_i:
                if p == cur_i:
                    continue
                cur_all_item.append(p)
            if len(cur_all_item) > 0:
                xx = cur_u, cur_i, opt_dis, cur_all_item
                cur_test_data.append(xx)
                for neg_i in neg_datas[0:10]:
                    if neg_i not in poi2xy:
                        continue
                    neg_x = poi2xy[neg_i][0]
                    neg_y = poi2xy[neg_i][1]
                    opt_dis = get_dis([poi_x, poi_y], [neg_x, neg_y])
                    xx = cur_u, neg_i, opt_dis, cur_all_item
                    cur_test_data.append(xx)
            ret_test.append(cur_test_data)

    return ret, user_num + 1, poi_num + 1, topic_k, ret_test


if __name__ == '__main__':
    train_data = process_fun("sample_data")
    print(len(train_data))
