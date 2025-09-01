import torch
import torch.nn as nn

# 定义一个简单 RNN
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# 模拟输入: batch=5, 序列长度=3, 特征维度=10
x = torch.randn(5, 3, 10)

# 初始化隐藏状态: (num_layers, batch, hidden_size)
h0 = torch.zeros(1, 5, 20)

# 前向传播
out, hn = rnn(x, h0)

print(out.shape)  # torch.Size([5, 3, 20]) 每个时间步的输出
print(hn.shape)   # torch.Size([1, 5, 20]) 最后时间步的隐藏状态
