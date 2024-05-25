import streamlit as st
import os
import json
import torch
import numpy as np
import pandas as pd
import docx2txt
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
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
        text_chunks = [text[i:i + (self.max_seq_len - 2)] for i in range(0, len(text), self.max_seq_len - 2)]
        all_entities = []

        for chunk in text_chunks:
            input_ids, attention_mask = self.ner_tokenizer(chunk)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            output = self.ner_model(input_ids, attention_mask)
            attention_mask = attention_mask.detach().cpu().numpy()
            length = sum(attention_mask[0])
            logits = output.logits
            logits = logits[0][1:length - 1]
            logits = [self.ner_id2label[i] for i in logits]
            entities = get_entities(logits)
            for ent in entities:
                ent_name = ent[0]
                ent_start = ent[1]
                ent_end = ent[2]
                all_entities.append((ent_name, "".join(chunk[ent_start:ent_end + 1])))
        
        result = {}
        for ent in all_entities:
            ent_name = ent[0]
            ent_text = ent[1]
            if ent_name not in result:
                result[ent_name] = [ent_text]
            else:
                result[ent_name].append(ent_text)
        
        return result

def convert_docx_to_txt(file):
    return docx2txt.process(file)

def convert_pdf_to_txt(file):
    try:
        fp = file
        text = ""
        parser = PDFParser(fp)
        doc = PDFDocument(parser)
        if not doc.is_extractable:
            return None
        resource = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(resource, laparams=laparams)
        interpreter = PDFPageInterpreter(resource, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            layout = device.get_result()
            for out in layout:
                if hasattr(out, "get_text"):
                    text += out.get_text()
        fp.close()
        return text
    except Exception as e:
        st.write(f"Error converting PDF: {e}")
        return None

def convert_txt_to_txt(file):
    return file.read().decode('utf-8')

def convert_file_to_text(uploaded_file):
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return convert_docx_to_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        return convert_pdf_to_txt(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return convert_txt_to_txt(uploaded_file)
    else:
        return None

# 使用Streamlit创建Web应用
def main():
    st.title("简历文本信息自动获取系统")
    st.write("作者: 韦简")
    st.write("[源代码链接](https://github.com/Antares28/Resume-Extraction)")

    st.write("请上传简历文件进行信息自动获取:")
    data_name = "resume"
    predictor = Predictor(data_name)

    uploaded_file = st.file_uploader("上传文件", type=["docx", "pdf", "txt"])
    if uploaded_file:
        input_text = convert_file_to_text(uploaded_file)
        if input_text:
            st.write("文件内容:")
            st.write(input_text)
            
            if st.button("识别"):
                ner_result = predictor.ner_predict(input_text)
                
                st.write("识别的标签与内容:")
                if ner_result:
                    result_df = pd.DataFrame([(k, v) for k, values in ner_result.items() for v in values], columns=["标签", "实体"])
                    st.dataframe(result_df)  # 使用st.dataframe而不是st.write来显示整个表格
                else:
                    st.write("未识别到实体。")
        else:
            st.write("文件格式不支持，请上传docx、pdf或txt文件")

if __name__ == "__main__":
    main()
