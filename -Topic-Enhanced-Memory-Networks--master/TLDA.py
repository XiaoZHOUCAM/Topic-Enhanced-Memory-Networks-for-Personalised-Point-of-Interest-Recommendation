# encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
'''
the demo of TLDA
input_data = [(doc_idx1,poi_idx1,time_idx1),(doc_idx2,poi_idx2,time_idx2),....]
Gibson sampling
'''
import numpy as np
import random
import pickle


class tlda:
    """
    @input_data:input_train_data
    @doc_index_list:index of the document
    @doc_len:length of each document
    @doc_num:number of documents
    @poi_num:number of POIs
    @time_num:number of time slots
    @topic_k:set k topics
    @iter:number of iterations
    @alpha:hyperparameter of pattern-user distribution
    @beta:hyperparameter of venue-pattern distribution
    @gamma:hyperparameter of pattern-time distribution
    """

    def __init__(self, input_data, doc_index, doc_len, doc_num, poi_num, time_num, topic_k, iter, alpha=0.1, beta=0.1,
                 gamma=10):
        self.input_data = input_data
        self.doc_len = doc_len
        self.doc_index = doc_index
        self.doc_num = doc_num
        self.poi_num = poi_num
        self.time_num = time_num
        self.topic_k = topic_k
        self.iter = iter
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.n_data = len(input_data)
        self.max_doc_len = max(self.doc_len)
        self._init_para()

    def _init_para(self):
        self.document_topic = np.zeros((self.doc_num, self.topic_k))
        self.time_topic = np.zeros((self.time_num, self.topic_k))
        self.topic_word = np.zeros((self.topic_k, self.poi_num))
        self.pre_topic_select = np.zeros((self.doc_num, self.max_doc_len), dtype=int)
        self.nwsum = np.zeros(self.topic_k, dtype="int")
        self.theta = np.zeros((self.doc_num, self.topic_k))
        self.phi = np.zeros((self.topic_k, self.poi_num))
        self.vt = np.zeros((self.time_num, self.topic_k))

    def _gen_result(self):
        for i in range(len(self.doc_index)):
            cur_doc_len = self.doc_len[i]
            cur_doc_index = self.doc_index[i]
            self.theta[cur_doc_index] = (self.document_topic[cur_doc_index] + self.alpha) / (
                        cur_doc_len + self.topic_k * self.alpha)
        for i in range(self.topic_k):
            self.phi[i] = (self.topic_word[i] + self.beta) / (self.nwsum[i] + self.poi_num * self.beta)
        for i in range(self.time_num):
            cur_time_len = sum(self.time_topic[i])
            self.vt[i] = (self.time_topic[i] + self.gamma) / (cur_time_len + self.topic_k * self.gamma)

    def Sample_topic(self, i, j, l, cur_len, pre_topic):
        self.document_topic[i, pre_topic] = self.document_topic[i, pre_topic] - 1
        self.time_topic[l, pre_topic] = self.time_topic[l, pre_topic] - 1
        self.topic_word[pre_topic, j] = self.topic_word[pre_topic, j] - 1
        self.nwsum[pre_topic] = self.nwsum[pre_topic] - 1
        cur_doc_topic = self.document_topic[i, :]
        cur_time_topic = self.time_topic[l, :]
        cur_topic_word = self.topic_word[:, j]
        Vbeta = self.poi_num * self.beta
        Kalpha = self.topic_k * self.alpha
        Kvt = self.topic_k * self.gamma
        cur_topic = -1
        cur_time_len = sum(self.time_topic[l])
        cur_topic_prob = (cur_topic_word + self.beta) / (self.nwsum + Vbeta) * \
                         (cur_doc_topic + self.alpha) / (cur_len + Kalpha) * \
                         (cur_time_topic + self.gamma) / (cur_time_len + Kvt)
        all_sum_t = sum(cur_topic_prob)
        cur_topic_prob = cur_topic_prob / all_sum_t
        for k in range(1, self.topic_k):
            cur_topic_prob[k] += cur_topic_prob[k - 1]
        u = random.uniform(0, cur_topic_prob[self.topic_k - 1])
        for topic in range(self.topic_k):
            if cur_topic_prob[topic] > u:
                cur_topic = topic
                break
        self.document_topic[i, cur_topic] = self.document_topic[i, cur_topic] + 1
        self.time_topic[l, cur_topic] = self.time_topic[l, cur_topic] + 1
        self.topic_word[cur_topic, j] = self.topic_word[cur_topic, j] + 1
        self.nwsum[cur_topic] = self.nwsum[cur_topic] + 1
        return cur_topic

    def train_model(self):
        # first random initialization topic
        for i in range(len(self.input_data)):
            for j in range(self.doc_len[i]):
                cur_u = self.input_data[i][j][0]
                cur_p = self.input_data[i][j][1]
                cur_t = self.input_data[i][j][2]
                topic = random.randint(0, self.topic_k - 1)
                self.document_topic[cur_u, topic] += 1
                self.time_topic[cur_t, topic] += 1
                self.topic_word[topic, cur_p] += 1
                self.nwsum[topic] += 1
                self.pre_topic_select[cur_u, j] = topic
        self._gen_result();

        for it in range(self.iter):
            print("iter:", it)
            for i in range(len(self.input_data)):
                for j in range(self.doc_len[i]):
                    cur_u = self.input_data[i][j][0]
                    cur_p = self.input_data[i][j][1]
                    cur_t = self.input_data[i][j][2]
                    cur_doc_len = self.doc_len[i]
                    cur_pre_topic = self.pre_topic_select[cur_u][j]
                    cur_topic = self.Sample_topic(cur_u, cur_p, cur_t, cur_doc_len, cur_pre_topic)
                    self.pre_topic_select[cur_u, j] = cur_topic
            self._gen_result();

    def save_model_data(self, input_file):
        print("save_model...")
        f_o = open(input_file, "wb")
        pickle.dump([self.theta, self.phi, self.vt], f_o)
        f_o.close()


if __name__ == '__main__':
    # program process
    tlda = TLDA(input_data, doc_index, doc_len, doc_num, poi_num, time_num, topic_k, iter);
    tlda.train_model()
    tlda.save_model_data("model_file")
