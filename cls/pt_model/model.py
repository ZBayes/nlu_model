from pt_model.transformer_cls import TransformerCls
from pt_model.textrcnn import TextRCNN
from pt_model.textcnn import TextCNN
from pt_model.textrnn import TextRNN
import os
from sklearn.metrics.classification import accuracy_score
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as torch_data_util
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
from tqdm import trange, tqdm
from torchsummary import summary
from loguru import logger

torch.set_printoptions(precision=8)

def parse_net_result(predict_result):
    scores = []
    labels = []
    for out in predict_result:
        out = out.detach().numpy()
        score = out[-1]
        label = np.where(out == max(out))[0][0]
        scores.append(score)
        labels.append(label)
    
    return labels, scores

class TrainModelPipeline():
    def __init__(self, config):
        self.config = config
        self.net = self.choose_model()
        logger.info(self.net)
        # summary(self.net, (64, 32))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def choose_model(self):
        logger.info("using model {}".format(self.config["experi_para"]["model_name"]))
        if self.config["experi_para"]["model_name"] == "TEXTCNN":
            return TextCNN(self.config)
        elif self.config["experi_para"]["model_name"] == "TEXTRNN":
            return TextRNN(self.config)
        elif self.config["experi_para"]["model_name"] == "TEXTRCNN":
            return TextRCNN(self.config)
        elif self.config["experi_para"]["model_name"] == "TRANSFORMERCLS":
            return TransformerCls(self.config)
        else:
            return TextCNN(self.config)

    def call_train(self, x_train, y_train):
        # 数据batch处理
        torch_dataset = cls_data(x_train, y_train)
        dataLoader = torch_data_util.DataLoader(
            dataset=torch_dataset,
            batch_size=self.config["train_para"]["batch_size"],             # 每批提取的数量
            shuffle=True,                                     # 要不要打乱数据（打乱比较好）
            num_workers=2                                     # 多少线程来读取数据
        )

        # 训练配置
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config["train_para"]["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        # 开始训练
        for epoch in range(self.config["train_para"]["epoch"]):
            with trange(len(list(dataLoader))) as t:
                for i, (sentences, labels) in enumerate(dataLoader):
                    t.set_description("EPOCH %s / %s" % (epoch + 1, self.config["train_para"]["epoch"]))
                    optimizer.zero_grad()
                    # sentences = torch.tensor(sentences).to(self.device)
                    # labels = torch.tensor(labels).to(self.device)
                    sentences = sentences.type(torch.LongTensor)
                    labels = labels.type(torch.LongTensor)
                    out = self.net(sentences)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()
                    
                    t.set_postfix(loss="%.8f"%loss.item(),batch_num=i + 1)
                    t.update(1)

    def call_evaluate(self, x_test, y_test, pred_path = ""):
        # 数据batch处理
        torch_dataset = cls_data(x_test, y_test)
        dataLoader = torch_data_util.DataLoader(
            dataset=torch_dataset,
            batch_size=self.config["train_para"]["batch_size"],             # 每批提取的数量
            shuffle=False,                                     # 要不要打乱数据（打乱比较好）
            num_workers=1                                     # 多少线程来读取数据
        )
        label_pres = []
        scores = []
        with trange(len(list(dataLoader))) as t:
            for i, (sentences, labels) in enumerate(dataLoader):
                predict = self.net(sentences)
                label_pre, score = parse_net_result(predict)
                label_pres.extend(label_pre)
                scores.extend(score)
                t.update(1)

        if self.config["train_para"]["class_num"] == 2:
            logger.info("auc")
            logger.info("%.8f"%roc_auc_score(y_test, scores))
        logger.info(y_test[:10])
        logger.info(scores[:10])
        logger.info("accuracy")
        logger.info(accuracy_score(y_test, label_pres))
        logger.info("confusion_matrix")
        confusion_matrix_get = confusion_matrix(y_test, label_pres)
        self.print_confusion_matrix(confusion_matrix_get)
        logger.info("classification_report")
        logger.info(classification_report(y_test, label_pres, digits=6))

        if pred_path != "":
            with open(pred_path, "w", encoding="utf8") as f:
                # 保存预测文件，方便进行分析：实际标签、预测标签、概率打分
                for idx in range(len(label_pres)):
                    f.write("{}\t{}\t{}%\n".format(y_test[idx], label_pres[idx], scores[idx]))
                
    
    def print_confusion_matrix(self, confusion_matrix):
        confusion_matrix_list = confusion_matrix.tolist()
        for line in confusion_matrix_list:
            logger.info(line)
    
    def save_model(self, path):
        path = path + "/model.pth"
        torch.save(self.net, path)
        logger.info("model saved: {}".format(path))



class cls_data(torch_data_util.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]