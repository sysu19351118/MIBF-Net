import torch
from torch import nn
from transformers import BertTokenizer, BertModel



class BertEncoder(nn.Module):
    def __init__(self,):  # num_labels 代表分类的数量
        super(BertEncoder, self).__init__()
        # 使用预训练的BERT模型
        self.bert = BertModel.from_pretrained('/mnt/sda1/zzixuantang/classfier_convNext/model/BERT_pretain')

    def forward(self, input_ids, attention_mask=None):
        # 获取BERT的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 获取[CLS]标记的表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output



if __name__ == "__main__":
    # 准备数据
    tokenizer = BertTokenizer.from_pretrained('/mnt/sda1/zzixuantang/classfier_convNext/model/BERT_pretain')
    text = ["I love machine learning", "I hate bugs in code"]

    # 使用Tokenizer将文本转化为模型可以处理的输入格式
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # 创建模型
    model = BertEncoder()  # 假设是二分类任务

    # 前向传播
    logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # 打印结果
    print(logits)
