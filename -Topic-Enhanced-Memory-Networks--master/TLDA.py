#encoding:utf-8
'''
the demo of TLDA
input_data = [(doc_idx1,poi_idx1,time_idx1),(doc_idx2,poi_idx2,time_idx2),....]
Gibson sampling
'''

import numpy as np
import random
import pickle
class 
:
    '''
    @input_data:input_train_data
    @doc_len: the list ( length of each docs)
    @doc_num:doc_num
    @poi_num:poi_num
    @time_num:time_num
    @topic_k: set k topics
    @iter: the iter num
    @alpha:doc-topic Hyperparameter
    @gamma:time-topic Hyperparameter
    @beta:topic-poi Hyperparameter
    '''
    def __init__(self, input_data,doc_len,doc_num,poi_num,time_num,topic_k,iter,alpha=0.1,gamma=0.1,beta=0.1):
        self.input_data = input_data
        self.doc_len = doc_len
        self.doc_num = doc_num
        self.poi_num = poi_num
        self.time_num = time_num
        self.topic_k = topic_k
        self.iter = iter
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.n_data = len(input_data)
        self._init_para(doc_num,poi_num,time_num,topic_k)
    
    def _init_para(self,doc_num,poi_num,time_num,topic_k):
        self.docment_topic = np.zeros((self.doc_num,self.topic_k))
        self.time_topic = np.zeros((self.time_num,self.topic_k))
        self.topic_word =  np.zeros((self.topic_k,self.poi_num))
        self.pre_topic_select = np.zeros(self.n_data,dtype=int)
        self.nwsum = np.zeros(self.topic_k,dtype="int")
        self.theta = np.zeros((self.doc_num,self.topic_k))
        self.phi = np.zeros((self.topic_k,self.poi_num))
        self.vt = np.zeros((self.time_num,self.topic_k))

    def _gen_result(self):
        for i in range(self.doc_num ):
            self.theta[i] = (self.docment_topic[i] + self.alpha)/( self.doc_len[i] +self.topic_k * self.alpha)
        for i in xrange(self.topic_k):
            self.phi[i] = (self.topic_word[i] + self.beta)/(self.nwsum[i] + self.poi_num*self.beta)
        for i in xrange(self.topic_k):
            self.vt.T[i] = (self.time_topic.T[i] + self.gamma)/(self.nwsum[i] + self.time_num*self.gamma)


    def Sample_topic(i,j,l,cur_len,pre_topic):
        self.docment_topic[i,pre_topic] = self.docment_topic[i,pre_topic] - 1
        self.time_topic[j,pre_topic] = self.time_topic[j,pre_topic] - 1
        self.topic_word[pre_topic,l] = self.topic_word[pre_topic,l] - 1
        self.nwsum[pre_topic] = self.nwsum[pre_topic] -1
        cur_doc_topic = self.docment_topic[i,:]
        cur_time_topic = self.time_topic[j,:]
        cur_topic_word = self.topic_word[:,l]
        Vbeta = self.poi_num*self.beta
        Kalpha = self.topic_k*self.alpha
        Kvt = self.topic_k*self.gamma
        cur_topic = -1
        cur_topic_prob = (cur_topic_word + self.beta)/(nwsum + Vbeta) * \
                    (cur_doc_topic + self.alpha)/(cur_len + Kalpha) * \
                    (cur_time_topic + self.gamma)/(nwsum + Kvt)
        for k in range(1,ar.K):
            cur_topic_prob[k] += cur_topic_prob[k-1]
        u = random.uniform(0,cur_topic_prob[ar.K-1])
        for topic in range(ar.K):
            if cur_topic_prob[topic] > u:
                cur_topic = topic
                break
        self.docment_topic[i,cur_topic] = self.docment_topic[i,cur_topic] + 1
        self.time_topic[j,cur_topic] = self.time_topic[j,cur_topic] + 1
        self.topic_word[cur_topic,l] = self.topic_word[cur_topic,l] + 1
        self.nwsum[cur_topic] = self.nwsum[cur_topic] +1
        return cur_topic

    def train_model(self):
        #first random initialization topic
        for j in range(self.n_data):
            cur_u = self.input_data[j][0]
            cur_p = self.input_data[j][1]
            cur_t = self.input_data[j][2]
            topic = random.randint(0,self.topic_k-1)
            self.docment_topic[cur_u,topic] += 1
            self.time_topic[cur_t,topic] += 1
            self.topic_word[topic,cur_p] += 1
            self.nwsum[topic] += 1
            self.pre_topic_select[j] = topic
        self.__gen_result();

        for it in range(self.iter):
            for j in range(self.n_data):
                cur_u = self.input_data[j][0]
                cur_p = self.input_data[j][1]
                cur_t = self.input_data[j][2]
                cur_doc_len = self.doc_len[cur_u]
                cur_pre_topic = self.pre_topic_select[j]
                cur_topic = Sample_topic(cur_u,cur_p,cur_t,cur_doc_len,cur_pre_topic)
                self.pre_topic_select[j] = cur_topic
            self.__gen_result();
    
    def save_model_data(self,input_file):
        f_o = open(input_file,"w")
        pickle.dump([ self.theta,self.phi,self.vt],f_o)
        f_o.close()


if __name__ == '__main__':
    #program process
    tlda = TLDA(input_data,doc_len,doc_num,poi_num,time_num,topic_k,iter);
    tlda.train_model()
    tlda.save_model_data("model_file")
        


    