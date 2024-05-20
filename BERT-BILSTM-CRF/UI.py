import streamlit as st
import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer

# 定义函数以获取模型参数
def get_args(args_path, args_name=None):
    with open(args_path, "r", encoding="utf-8") as fp:
        args_dict = json.load(fp)
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args

# 定义预测类
class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin"), map_location="cpu"))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result

# 使用Streamlit创建Web应用
def main():
    st.title("命名实体识别(NER)系统")
    st.write("请输入文本进行命名实体识别:")

    data_name = "resume"
    predictor = Predictor(data_name)

    input_text = st.text_area("输入文本")
    if st.button("识别"):
        if input_text:
            ner_result = predictor.ner_predict(input_text)
            st.write("文本: ", input_text)
            st.write("识别的实体: ", ner_result)
        else:
            st.write("请输入文本")

if __name__ == "__main__":
    main()
