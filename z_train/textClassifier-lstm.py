import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import jieba
from torch.utils.data import Dataset, DataLoader
'''
使用word2vec + lstm 进行文本分类
'''

data = pd.read_csv('./z_train/dataset.csv',sep='\t',header=None)
lables = data[1].to_list()
lable_to_index = {lable : i for i,lable in enumerate(set(lables))}
index_to_label = {idx: label for label, idx in lable_to_index.items()}

texts = data[0].to_list()

sentences = [jieba.lcut(text) for text in texts]

word2vec = Word2Vec(
    sentences=sentences, #训练语料
    vector_size=100, # 词向量维度
    window=5,        # 窗口大小:当前词与预测词之间的最大距离
    min_count=1,     # 最小词频
    sg=1             # 1:skip-gram, 0:CBOW
)

SEQ_LEN = 30

# 将分词序列转换为固定长度的嵌入序列（不做平均）
def tokens_to_sequence(tokens, w2v_model, seq_len=SEQ_LEN):
    emb_size = w2v_model.vector_size
    sequence = []
    for token in tokens:
        if token in w2v_model.wv:
            sequence.append(w2v_model.wv[token])
        else:
            sequence.append(np.zeros(emb_size, dtype=np.float32))
        if len(sequence) == seq_len:
            break
    # padding 到固定长度
    while len(sequence) < seq_len:
        sequence.append(np.zeros(emb_size, dtype=np.float32))
    return np.stack(sequence, axis=0).astype(np.float32) # 形状: (30, 100)

class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, w2v_model, seq_len=SEQ_LEN):
        self.tokenized_texts = tokenized_texts
        self.label_indices = [lable_to_index[lbl] for lbl in labels]
        self.w2v_model = w2v_model
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        seq = tokens_to_sequence(tokens, self.w2v_model, self.seq_len)
        x_tensor = torch.tensor(seq, dtype=torch.float32) # 形状: (30, 100)
        y_tensor = torch.tensor(self.label_indices[idx], dtype=torch.long) # 标量
        return x_tensor, y_tensor

# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 输入张量形状: (批量, 30, 100)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # 形状: (2, 批量, 128)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # 形状: (2, 批量, 128)
        out, _ = self.lstm(x, (h0, c0)) # 输出: (批量, 30, 128)
        out = self.fc(out[:, -1, :])  # 取最后时刻: (批量, 128) -> (批量, 类别数)
        return out

# 模型参数
input_size = 100  # word2vec向量维度
hidden_size = 128
num_layers = 2
num_classes = len(lable_to_index)

dataset = TextDataset(sentences, lables, word2vec, seq_len=SEQ_LEN)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)   # 训练批: (32, 30, 100)
eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)   # 评估批: (64, 30, 100)

# 创建模型
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss() # 输入: (批量, 类别数); 目标: (批量,)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
num_epochs = 50

# 训练循环
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y in train_loader:  # 形状: (32, 30, 100), (32,)
        optimizer.zero_grad()
        outputs = model(batch_x)            # 形状: (32, 类别数)
        loss = criterion(outputs, batch_y)  # 标量
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        total_count += batch_x.size(0)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/max(1,total_count):.4f}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in eval_loader:   # 形状: (64, 30, 100), (64,)
        outputs = model(batch_x)           # 形状: (64, 类别数)
        _, predicted = torch.max(outputs.data, 1) # 形状: (64,)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
accuracy = correct / max(1, total)
print(f'测试准确率: {accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'lstm_classifier.pth')
print('模型已保存为 lstm_classifier.pth')

# 预测函数
def predict_text(text, model, w2v_model, index_to_label, seq_len=SEQ_LEN):
    tokens = jieba.lcut(text)
    seq = tokens_to_sequence(tokens, w2v_model, seq_len)
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) # 形状: (1, 30, 100)
    model.eval()
    with torch.no_grad():
        output = model(x)                  # 形状: (1, 类别数)
        _, predicted = torch.max(output, 1) # 形状: (1,)
        predicted_label = index_to_label[predicted.item()]
        confidence = torch.softmax(output, dim=1).max().item()
    return predicted_label, confidence

# 测试文本预测
print("\n=== 测试文本预测 ===")
test_texts = [
    "导航到上海",
    "帮我定明天9点的闹钟",
    "帮我播放最新的新闻",
    "打开微信"
]

for text in test_texts:
    predicted_label, confidence = predict_text(text, model, word2vec, index_to_label)
    print(f"文本: {text}")
    print(f"预测标签: {predicted_label}")
    print(f"置信度: {confidence:.4f}")
    print("-" * 50)



