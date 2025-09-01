import torch
import torch.nn as nn
import torch.optim as optim
from typing import List,Dict
import pandas as pd

'''
使用 词袋向量 + fnn前馈神经网络 实现文本分类
'''

#数据准备
data = pd.read_csv('./z_train/dataset.csv', encoding='utf-8', sep='\t',header=None)
text_list = data[0].to_list()
lable_list = data[1].to_list()

#标签数字化
lable_to_index = {lable : i for i,lable in enumerate(set(lable_list))}
index_to_lable = {i : lable for lable,i in lable_to_index.items()}
print('标签:\n',lable_to_index)
lables = [lable_to_index[lable] for lable in lable_list]

#定义词典，将data中text每个字符映射到数字
char_to_indexs = {'<bag>':0}
for text in text_list:
    for char in text:
        if char not in char_to_indexs:
            char_to_indexs[char] = len(char_to_indexs)
print('词典:\n',char_to_indexs)

#文本数字化
max_len = 60
tokenied_texts = []
for text in text_list:
    tokenied_text = [char_to_indexs.get(char,0) for char in text[:max_len]]
    tokenied_text += [0] * (max_len - len(tokenied_text)) #不足60补0
    tokenied_texts.append(tokenied_text)
print('文本数字化:\n',tokenied_texts[:10][:]) #list of tensor  12100,60

#标签转向量
lables = torch.tensor(lables)
print('标签向量:\n',lables)
print('标签向量形状:\n',lables.shape) #tensor(12100)

#词袋bag of word：term_frequency 记录每个词出现的次数 [12100,词典大小]
#文本列表转换为词袋向量
def create_bow_vector(tokenied_texts:List[List[int]],char_to_indexs:Dict)->torch.Tensor:
    bow_vectors = []
    for tokenied_text in tokenied_texts:
        bow_vector = torch.zeros(len(char_to_indexs)) #词典大小，初始化为0
        for index in tokenied_text:
            if index != 0:
                bow_vector[index] += 1
        bow_vectors.append(bow_vector)
    return torch.stack(bow_vectors)

bow_vectors = create_bow_vector(tokenied_texts,char_to_indexs)
print(f'词袋向量:\n {bow_vectors[0:5][:]},形状:{bow_vectors.shape}')

#模型定义
class FNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(FNN,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return output

input_size = len(char_to_indexs) #输入维度就是词典大小
hidden_size = 100
output_size = len(lable_to_index) #输出维度就是标签个数

model = FNN(input_size,hidden_size,output_size)
criterion = nn.CrossEntropyLoss() #分类损失函数
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs = 1000
#训练
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad() #清空梯度
    #前向传播
    output = model(bow_vectors)
    loss = criterion(output,lables)
    #反向传播
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch:{epoch},loss:{loss.item()}')

def text_classify(text,model,char_to_indexs,max_len,index_to_lable):
    tokenied_text = [char_to_indexs.get(char,0) for char in text[:max_len]]
    tokenied_text += [0] * (max_len - len(tokenied_text))

    bow_vector = torch.zeros(len(char_to_indexs))#词典大小，初始化为0
    for index in tokenied_text:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = torch.tensor(bow_vector).unsqueeze(0)
    print(f'词袋向量:{bow_vector}')

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)
        print(f'输出:{output}')
        _,predicted = torch.max(output,1)
        print(f'预测标签:{predicted.item()}')
        return index_to_lable[predicted.item()]

query_text = "我要订个明天早上7点50的闹钟"
result = text_classify(query_text,model,char_to_indexs,max_len,index_to_lable)
print(f'query_text:{query_text},result:{result}')

