import numpy as np
import datetime
import os
from loguru import logger
from gensim.models import KeyedVectors

class init_experiment():
    def __init__(self, config):
        self.config = config
        self.__init__folder()
    
    def __init__folder(self):
        self.build_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.exeri_path = "./shopping_reviews/data/result_report/" + self.config["experi_para"]["model_name"] + "_" + self.build_time
        os.makedirs(self.exeri_path)
        logger.add(self.exeri_path+"/experiment.log")
        logger.info("base experiment space builded: {}".format(self.exeri_path))

class word2vector_model():
    def load_pretrained(self, ptm_path):
        # 加载
        # w2v_model = KeyedVectors.load_word2vec_format("./shopping_reviews/data/embedding/sgns.renmin.bigram.w2v")
        w2v_model = KeyedVectors.load_word2vec_format(ptm_path) # 基本gensim对象

        self.N_DIM = len(w2v_model[list(w2v_model.wv.vocab.keys())[0]]) # 词向量维数

        # word2vec后处理
        n_symbols = len(w2v_model.wv.vocab.keys()) + 2
        embedding_weights = [[0 for i in range(self.N_DIM)] for i in range(n_symbols)]
        np.zeros((n_symbols, 300))
        idx = 1
        word2idx_dic = {}
        w2v_model_metric = []
        for w in w2v_model.wv.vocab.keys():
            embedding_weights[idx] = w2v_model[w]
            word2idx_dic[w] = idx
            idx = idx + 1

        # 留给未登录词的位置
        avg_weights = [0 for i in range(self.N_DIM)]
        for wd in word2idx_dic:
            avg_weights = [(avg_weights[idx]+embedding_weights[word2idx_dic[wd]][idx]) for idx in range(self.N_DIM)]
        avg_weights = [avg_weights[idx] / len(word2idx_dic) for idx in range(self.N_DIM)]
        embedding_weights[idx] = avg_weights
        word2idx_dic["<UNK>"] = idx

        # 留给pad的位置
        word2idx_dic["<PAD>"] = 0

        self.word2idx_dic = word2idx_dic
        self.embedding_weights = embedding_weights
        self.n_symbols = n_symbols

    def word2idx(self, word):
        if len(self.word2idx_dic) == 0:
            self.__load_default__()
        if word in self.word2idx_dic:
            return self.word2idx_dic[word]
        else:
            return len(self.word2idx_dic) - 1

    def sentence2idx(self, sentence, batch_len = -1):
        sentence_idx = []
        for idx in range(len(sentence)):
            sentence_idx.append(self.word2idx(sentence[idx]))
        if batch_len >= 0:
            if len(sentence_idx) > batch_len:
                sentence_idx = sentence_idx[:batch_len]
            else:
                while len(sentence_idx) < batch_len:
                    sentence_idx.append(0)
        return sentence_idx

    def batch2idx(self, source_data, batch_len = -1):
        result_data = []
        for idx in range(len(source_data)):
            result_data.append(np.array(self.sentence2idx(source_data[idx], batch_len)))
        return result_data