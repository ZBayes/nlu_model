import re
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def cw(x): 
    # 基本的预处理操作
    punctuation = r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：]+"
    x = re.sub(punctuation, "", x)

    return list(jieba.cut(x))

def loadfile():
    # 加载并预处理模型
    neg = pd.read_excel('./shopping_reviews/data/source_data/neg.xls', header=None, index=None)
    pos = pd.read_excel('./shopping_reviews/data/source_data/pos.xls', header=None, index=None)

    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(
        np.concatenate((pos['words'], neg['words'])), y, test_size=0.2,random_state=666)
    
    return x_train, x_test, y_train, y_test