import torch
import torch.nn as nn
import numpy as np

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ProbSparseSelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(queries.size(-1))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, values)

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(InformerEncoderLayer, self).__init__()
        self.self_attn = ProbSparseSelfAttention(dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))
        residual2 = x
        x = self.norm2(residual2 + self.dropout(self.fc2(self.activation(self.fc1(x)))))
        return x

class InformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class InformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, d_ff=128, num_layers=2, dropout=0.1, forecast_steps=10):
        super(InformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = InformerEncoder(encoder_layer, num_layers)
        self.projection = nn.Linear(d_model, 1)
        self.forecast_steps = forecast_steps

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.projection(x)
        return x[:, -self.forecast_steps:, :]
