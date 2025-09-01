import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist() # 数据集第一列 文本
string_labels = dataset[1].tolist() # 数据集第二列 类别

# 类别转换数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 原始的文本构建一个词典，字 -》 数字
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 取文本的前40个字符
max_len = 40

tokenized_texts = []
for text in texts:
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    tokenized_texts.append(tokenized)

labels_tensor = torch.tensor(numerical_labels, dtype=torch.long)

# term frequency
def create_bow_vectors(tokenized_texts, vocab_size):
    bow_vectors = []
    for text_indices in tokenized_texts:
        bow_vector = torch.zeros(vocab_size) # 词典个数长度的向量，存储每个字在这个文本中间出现的次数
        for index in text_indices:
            if index != 0:  # Ignore padding
                bow_vector[index] += 1
        bow_vectors.append(bow_vector)
    return torch.stack(bow_vectors)

bow_matrix = create_bow_vectors(tokenized_texts, vocab_size)
input_size = vocab_size

# 实际项目中，优先考虑迁移学习，复用已有的网络网络，qwen
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

hidden_dim = 128
output_dim = len(label_to_index)
model = SimpleClassifier(input_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss() # 分类损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01) # Adam 优化器  可以结合梯度 动态调整学习， 0.01 -> 0.001 -> 0.00001

for _ in range(10):
    model.train()
    optimizer.zero_grad()

    outputs = model(bow_matrix)
    loss = criterion(outputs, labels_tensor)

    loss.backward()
    optimizer.step()

    print(f"Training complete. Loss: {loss.item():.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    # 正向传播，11 神经元的输出
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")