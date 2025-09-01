import torch
import torch.nn as nn # 深度学习的搭建

# 设置参数
input_size = 10
hidden_size = 20
sequence_length = 10

# 初始化 RNN 模型
rnn = nn.RNN(input_size, hidden_size, batch_first=True)

# 准备输入数据
# batch_size = 1, sequence_length = 1, input_size = 10
x = torch.randn(1, sequence_length, input_size)

# 初始化隐藏状态
h0 = torch.zeros(1, 1, hidden_size)

# 前向传播
# PyTorch 的 RNN 会自动处理所有时间步
output, hn = rnn(x, h0)
print("PyTorch RNN 模型的输出 (h_1):")
print(output)
print("PyTorch RNN 模型的最终隐藏状态 (hn):")
print(hn)
print("-" * 50)

# 5. 手动计算验证
# 获取模型参数
W_ih = rnn.weight_ih_l0
W_hh = rnn.weight_hh_l0
b_ih = rnn.bias_ih_l0
b_hh = rnn.bias_hh_l0

# 理论计算
# h_1 = tanh(W_ih * x_1 + b_ih + W_hh * h_0 + b_hh)
# PyTorch 默认输入是 (seq_len, batch, input_size)，这里我们使用了 batch_first=True，所以输入是 (batch, seq_len, input_size)
x1_squeeze = x.squeeze(1) # 移除 sequence_length 维度，变为 (1, 10)
h0_squeeze = h0.squeeze(0) # 移除 num_layers 维度，变为 (1, 20)
h1_manual = torch.tanh(torch.matmul(x1_squeeze, W_ih.t()) + b_ih + torch.matmul(h0_squeeze, W_hh.t()) + b_hh)

print("手动计算的 RNN 隐藏状态 (h_1):")
print(h1_manual)
print("-" * 50)

# 验证结果是否一致
print("手动计算与 PyTorch 模型输出是否接近：")
print(torch.allclose(output, h1_manual))