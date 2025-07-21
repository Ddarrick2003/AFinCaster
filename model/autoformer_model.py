import torch
import torch.nn as nn
import numpy as np

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(AutoformerEncoderLayer, self).__init__()
        self.decomp = SeriesDecomposition()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        attn_output, _ = self.self_attn(seasonal, seasonal, seasonal)
        x = self.norm(seasonal + self.dropout(attn_output))
        x = self.norm(x + self.dropout(self.linear2(self.activation(self.linear1(x)))))
        return x + trend  # trend added back

class AutoformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, d_ff=128, num_layers=2, dropout=0.1, forecast_steps=10):
        super(AutoformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder = nn.ModuleList([AutoformerEncoderLayer(d_model, d_ff, dropout) for _ in range(num_layers)])
        self.projection = nn.Linear(d_model, 1)
        self.forecast_steps = forecast_steps

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.projection(x)
        return x[:, -self.forecast_steps:, :]
