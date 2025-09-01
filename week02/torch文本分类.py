import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

dataset = pd.read_csv("./week01/dataset.csv", sep="\t", header=None)
print(dataset.head(10))
#文本 标签
texts = dataset[0].to_list()
string_lables = dataset[1].to_list()

#标签数字化
lable_to_index = {lable : i for i,lable in enumerate(set(string_lables))}
lables = [lable_to_index[lable] for lable in string_lables]
print("\n标签索引:",lable_to_index)

#文本构建一个词典  字 -> 数字
char_to_index = {'<pad>':0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
print(f"\n词典:",char_to_index)
#词典大小
vocab_size = len(char_to_index)

#获取转换为数字后的文本列表
max_len = 40
tokenized_texts = []
for text in texts:
    tokenized_text = [char_to_index.get(char,0) for char in text[:max_len]]
    tokenized_text += [0] * (max_len - len(tokenized_text)) #不足40补0
    tokenized_texts.append(tokenized_text)
print(f"\n文本列表：\n",tokenized_texts[0:5][:])

#标签转向量
lables_tensor = torch.tensor(lables,dtype=torch.long)

#词袋模型(bag of words)的向量化函数，将文本转换为词频向量
#输入参数1:已经分词并转换为索引的文本列表，每个文本表示为一个索引列表
#输入参数2:词典的大小（即多少个字），即向量的维度
def create_bow_vectors(tokenized_texts,vocab_size):
    bow_vectors = []
    for text_indexs in tokenized_texts:
        bow_vector = torch.zeros(vocab_size) #词典大小的0向量,存储每个字在当前文本出现的次数
        for index in text_indexs:
            if index != 0 :
                bow_vector[index] += 1
        bow_vectors.append(bow_vector)
    return torch.stack(bow_vectors) #将多个形状相同的张量合并成一个更高维度的张量

#输入词袋模型矩阵
bow_matrix = create_bow_vectors(tokenized_texts,vocab_size)
print(f"\n文本词袋模型矩阵:\n",bow_matrix[0:5][:])
print(f"文本词袋模型矩阵形状:\n",bow_matrix.shape)
print("\n")
input_size = vocab_size #输入的维度就是词典大小


#全链接网络
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
hidden_dim = 128
output_dim = len(lable_to_index) #输出维度为类别个数

model = SimpleClassifier(input_size,hidden_dim,output_dim)
criterion = nn.CrossEntropyLoss() #分类损失函数
optimizer = optim.SGD(model.parameters(),lr=0.02) #优化器

for _ in range(1000):
    model.train() #开启训练
    optimizer.zero_grad() #清空梯度
    #正向传播
    outputs = model.forward(bow_matrix)
    #计算损失
    loss = criterion(outputs,lables_tensor)
    #反向传播和优化
    loss.backward() #计算梯度
    optimizer.step() #更新参数

    print(f"####loss:{loss.item()}####")

#文本分类.  待预测文本,模型,词典,词典大小,文本最大长度,标签张量
def classify_text(text,model,char_to_index,vocab_size,max_len,index_to_lable):
    #将待预测文本转张量
    tokenized = [char_to_index.get(char,0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0 :
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0) #在第0维即最外层增加一个维度
    print("\n输入张量：",bow_vector)

    #正向传播
    model.eval() #将模型设置为评估模式
    with torch.no_grad(): #不追踪梯度
        output = model.forward(bow_vector) #输出一般为一个包含每个类别预测得分的张量
        print("\n输出:",output)
    
    _,predicted_index = torch.max(output,1) #在第1维找最大值
    predicted_index = predicted_index.item()
    print("\n得分最大值index=",predicted_index)
    predicted_lable = index_to_lable[predicted_index]

    return predicted_lable

index_to_lable = {i : lable for lable,i in lable_to_index.items()}

query_text = "帮我导航到北京"
predicted_class = classify_text(query_text,model,char_to_index,vocab_size,max_len,index_to_lable)
print(f"输入:{query_text} 预测为:{predicted_class}")
