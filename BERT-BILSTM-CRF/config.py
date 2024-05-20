import os
import torch
import json
from transformers import BertTokenizer


class CommonConfig:
    bert_dir = "./model_hub/chinese-bert-wwm-ext/" # 存储BERT模型的目录路径
    output_dir = "./checkpoint/" # 存储输出结果的目录路径
    data_dir = "./data/" # 数据集的目录路径


class NerConfig:
    def __init__(self, data_name):
        # 初始化通用配置
        cf = CommonConfig()
        self.bert_dir = cf.bert_dir
        self.output_dir = cf.output_dir
        
        # 设置输出目录
        self.output_dir = os.path.join(self.output_dir, data_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)    
        self.data_dir = cf.data_dir

        # 读取数据路径和标签
        self.data_path = os.path.join(self.data_dir, data_name)
        with open(os.path.join(self.data_path, "labels.txt"), "r", encoding="utf-8") as fp:
            self.labels = fp.read().strip().split("\n")
            
        # 生成BIO格式的标签和标签映射
        self.bio_labels = ["O"]
        for label in self.labels:
            self.bio_labels.append("B-{}".format(label))
            self.bio_labels.append("I-{}".format(label))
        print(self.bio_labels)
        self.num_labels = len(self.bio_labels)
        self.label2id = {label: i for i, label in enumerate(self.bio_labels)}
        print(self.label2id)
        self.id2label = {i: label for i, label in enumerate(self.bio_labels)}

        # 其他训练配置参数
        self.max_seq_len = 128          # 最大序列长度
        self.epochs = 3                 # 训练轮数
        self.train_batch_size = 12      # 训练时的批处理大小
        self.dev_batch_size = 1         # 验证时的批处理大小
        self.bert_learning_rate = 3e-5  # BERT模型的学习率
        self.crf_learning_rate = 1e-4   # CRF的学习率
        self.adam_epsilon = 1e-8        # Adam优化器的epsilon值
        self.weight_decay = 0.01        # 权重衰减
        self.warmup_proportion = 0.01   # 预热比例
        self.save_step = 500            # 模型保存的步数
