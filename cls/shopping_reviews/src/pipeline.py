# pipline

import json
from loguru import logger
from pt_model.model import TrainModelPipeline
from shopping_reviews.src.dataloader import loadfile
from shopping_reviews.src.util import word2vector_model, init_experiment

if __name__ == "__main__":
    # 配置加载
    config = json.load(open("./shopping_reviews/config/textcnn_base_config.json"))

    # 实验配置初始化
    experiment_space = init_experiment(config)
    logger.info("expriment description: {}".format(config["experi_para"]["description"]))

    # 数据加载
    x_train, x_test, y_train, y_test = loadfile()
    logger.info("dataset loaded，training set：{}，testing set：{}".format(len(x_train), len(x_test)))

    # 模型加载
    word2vector_model = word2vector_model()
    word2vector_model.load_pretrained(config["embedding"]["embedding_path"])
    config["embedding"]["word2vector_model"] = word2vector_model
    config["embedding"]["vocab_size"] = word2vector_model.n_symbols
    config["embedding"]["embed_dim"] = word2vector_model.N_DIM
    logger.info("word2vector loaded，amount of vocab: {}, dim: {}".format(word2vector_model.n_symbols, word2vector_model.N_DIM))

    # 数据预处理转换
    x_train = word2vector_model.batch2idx(x_train, batch_len=config["train_para"]["max_len"])
    x_test = word2vector_model.batch2idx(x_test, batch_len=config["train_para"]["max_len"])
    
    # 结果展示
    train_model_pipeline = TrainModelPipeline(config)
    train_model_pipeline.call_train(x_train, y_train)
    train_model_pipeline.call_evaluate(x_train, y_train)
    train_model_pipeline.call_evaluate(x_test, y_test, pred_path = experiment_space.exeri_path + "/test_set.tsv")
    train_model_pipeline.save_model(experiment_space.exeri_path)