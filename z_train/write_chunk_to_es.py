from transformers import BertTokenizer, BertModel
import torch

from elasticsearch import Elasticsearch

x = ['机器学习是一门研究如何让计算机从数据中学习的学科。']

# 加载预训练模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
model = BertModel.from_pretrained('./models/google-bert/bert-base-chinese')

# 将文本转换为模型输入
encoding = tokenizer(x, truncation=True, padding=True, max_length=64, return_tensors='pt')

# 获取文本向量
with torch.no_grad():
    outputs = model(**encoding)
    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state
    # 使用[CLS]标记的隐藏状态作为句子向量
    sentence_vector = last_hidden_states[:, 0, :].numpy()

print("句子向量:", sentence_vector)
print("向量维度:", sentence_vector.shape)

# 连接Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# 准备文档数据
doc = {
    "chunk_id": "c001",
    "document_id": 1,
    "knowledge_id": "ml001",
    "page_number": 1,
    "chunk_content": "机器学习是一门研究如何让计算机从数据中学习的学科。",
    "chunk_images": "/images/ml001_page1.png",
    "chunk_tables": "/tables/ml001_page1.csv",
    "embedding_vector": sentence_vector[0].tolist()  # 将numpy数组转换为列表
}

# 将文档写入Elasticsearch
response = es.index(index="chunk_info", body=doc)
print("文档已写入Elasticsearch:", response)