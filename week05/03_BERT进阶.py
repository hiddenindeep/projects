from transformers import AutoTokenizer, BertModel
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("../models/google-bert/bert-base-chinese/")
model = BertModel.from_pretrained("../models/google-bert/bert-base-chinese/")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt") # 构建输入
outputs = model(**inputs) # 输出层

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
print(outputs.pooler_output.shape, outputs.last_hidden_state.shape)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

pooling = MeanPooling()
res = pooling(
    outputs.last_hidden_state,
    inputs['attention_mask']
)
print("MeanPooling", res.shape)


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings

pooling = MaxPooling()
res = pooling(
    outputs.last_hidden_state,
    inputs['attention_mask']
)
print("MaxPooling", res.shape)


class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings

pooling = MinPooling()
res = pooling(
    outputs.last_hidden_state,
    inputs['attention_mask']
)
print("MinPooling", res.shape)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

pooling = AttentionPooling(768)
res = pooling(
    outputs.last_hidden_state,
    inputs['attention_mask']
)
print("WeightedLayerPooling", res.shape)


class CustomModel(nn.Module):
    def __init__(self, dropout=0.5, hidden_size=768, ner_labels=2, cls_labels=2):
        super().__init__()

        self.model = BertModel.from_pretrained("../models/google-bert/bert-base-chinese/", add_pooling_layer=True)
        self.dropout = nn.Dropout(dropout)
        self.ner_classifier = nn.Linear(hidden_size, ner_labels)
        self.cls_classifier = nn.Linear(hidden_size, cls_labels)

    def forward(self, inputs):
        x = self.model(**inputs)  # N * 768
        last_hidden_state = self.dropout(x.last_hidden_state)  # N * 768
        pooler_output = self.dropout(x.pooler_output)  # 768

        return self.ner_classifier(last_hidden_state), self.cls_classifier(pooler_output)

model = CustomModel()
feat = model(inputs)
print(feat)
