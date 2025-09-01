import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为词嵌入添加位置信息。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够大的位置编码矩阵，(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引 (0, 1, 2, ...)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算不同维度的除数，使用对数空间以提高数值稳定性
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 应用正弦函数到偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        # 应用余弦函数到奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加一个批次维度，使其形状为 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 将pe注册为模型的一个缓冲区（buffer），它不会被视为模型参数，
        # 但会随着模型保存和加载。
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x的形状应为 (batch_size, sequence_length, d_model)
        """
        # 将位置编码加到输入中，只取与输入序列长度匹配的部分
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块。
    """

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义 Q, K, V 的线性投影层
        # 这些层将输入投影到 Q, K, V 空间
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # 最后的输出线性层
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value 的形状均为 (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 1. 对输入进行线性投影，并分割成多个头
        # 投影后形状: (batch_size, seq_len, d_model)
        # 变形后形状: (batch_size, seq_len, num_heads, d_k)
        # 转置后形状: (batch_size, num_heads, seq_len, d_k)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力得分
        # 注意力得分 = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 应用掩码（mask），如果提供的话
        if mask is not None:
            # 将掩码中值为0的位置设置为一个非常小的负数，这样在softmax后这些位置的权重接近0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. 对得分进行softmax，得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. 用注意力权重加权 V
        context = torch.matmul(attention_weights, v)

        # 6. 将多个头的输出拼接，并再次进行线性投影
        # 形状变回 (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_linear(context)

        return output


# --- 完整示例：一个简单的Transformer编码器块 ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力机制：Q, K, V 都来自同一个输入 x
        attn_output = self.mha(x, x, x)
        # 第一个残差连接和层归一化
        out1 = self.norm1(x + attn_output)

        # 前馈网络
        ffn_output = self.ffn(out1)
        # 第二个残差连接和层归一化
        out2 = self.norm2(out1 + ffn_output)

        return out2


# 假设我们有一个词汇表，大小为1000
vocab_size = 1000
# 词嵌入的维度
d_model = 512
# 序列的最大长度
max_len = 100
# 注意力头的数量
num_heads = 8
# 批次大小
batch_size = 16
# 模拟输入序列
input_sequence = torch.randint(0, vocab_size, (batch_size, max_len))

# 实例化模型组件
embedding = nn.Embedding(vocab_size, d_model)
pos_encoder = PositionalEncoding(d_model, max_len)
transformer_block = TransformerBlock(d_model, num_heads)

print("--- 准备输入数据 ---")
print(f"输入序列形状: {input_sequence.shape}")

# 1. 词嵌入
embedded_x = embedding(input_sequence)
print(f"词嵌入后形状: {embedded_x.shape}")

# 2. 添加位置编码
# 这就是 Q, K, V 的来源（对于自注意力而言）
input_with_pos_encoding = pos_encoder(embedded_x)
print(f"添加位置编码后形状: {input_with_pos_encoding.shape}")

# 3. 将其送入多头注意力块
print("\n--- 运行 TransformerBlock ---")
output = transformer_block(input_with_pos_encoding)

print(f"输出形状: {output.shape}")
print("代码成功运行！")