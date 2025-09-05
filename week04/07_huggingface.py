import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# pip install modelscope
# 下载命令 modelscope download --model google-bert/bert-base-chinese --local_dir ../models/google-bert/bert-base-chinese
# https://www.modelscope.cn/models/google-bert/bert-base-chinese/

#分词器，模型结构，权重
tokenizer = AutoTokenizer.from_pretrained("./models/google-bert/bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("./models/google-bert/bert-base-chinese")

# 待处理的文本
text = "人工智能正在[MASK]改变我们的生活。"

# 使用分词器对文本进行编码
encoded_input = tokenizer(text, return_tensors='pt')
print("编码后的输入张量：")
print(encoded_input)
print("-----------------------")
'''
{'input_ids': tensor([[ 101,  782, 2339, 3255, 5543, 3633, 1762,  103, 3121, 1359, 2769,  812, 4638, 4495, 3833,  511,  102]]), 字在词典的位置
  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),  句子类型，0表示第一句，1表示第二句
  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) 注意力掩码，1表示需要关注，0表示不需要关注
}
'''

# 打印分词结果
tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
print("分词结果：")
print(tokens)


# 增加一个文本示例
text_to_feature_extraction = "自然语言处理是人工智能的一个重要分支。"

# 对文本进行编码
encoded_input_features = tokenizer(text_to_feature_extraction, return_tensors='pt')

# 将编码后的输入传递给模型，禁用梯度计算以节省内存和提高速度
with torch.no_grad():
    outputs = model(**encoded_input_features, output_hidden_states=True)

# outputs是一个包含多个元素的元组或对象
# 第一个元素是 logits (用于完形填空任务)，第二个元素是隐藏层状态
# outputs.hidden_states 包含了所有层的隐藏层输出，是一个元组
# 最后一个元素是最后一层的输出，倒数第二个是倒数第二层的输出
last_hidden_state = outputs.hidden_states[-1]
# 也可以访问倒数第二层
second_to_last_hidden_state = outputs.hidden_states[-2]

print("文本的token ID数量:", encoded_input_features['input_ids'].shape[1])
print("最后一层隐藏层输出的形状:", last_hidden_state.shape)
print("倒数第二层隐藏层输出的形状:", second_to_last_hidden_state.shape)

# 隐藏层输出的形状为：[batch_size, sequence_length, hidden_size]
# last_hidden_state[0] 代表批次中的第一个（也是唯一一个）样本
# last_hidden_state[0][0] 代表 [CLS] token 的向量表示
cls_embedding = last_hidden_state[0][0]
print("[CLS] token 的向量表示形状:", cls_embedding.shape)

# 获取整个序列的特征向量（例如，取所有 token 向量的平均值）
# 这是一个简单的池化策略
mean_pooling_embedding = torch.mean(last_hidden_state[0], dim=0)
print("通过均值池化得到的序列特征向量形状:", mean_pooling_embedding.shape)