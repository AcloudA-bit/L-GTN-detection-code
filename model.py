import torch
from torch import nn
import torch.nn.functional as F


class ConvAttention(nn.Module):
    """Pointwise convolutional attention, normalized along the sequence."""

    def __init__(self, d_model):
        super().__init__()
        self.conv_attn = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_value = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        attention = F.softmax(self.conv_attn(x), dim=-1)
        values = self.conv_value(x)
        return self.proj_out((attention * values).transpose(1, 2))


class ConvFFN(nn.Module):
    """Depthwise convolution followed by pointwise expansion and projection."""

    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.conv_depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        self.conv_pointwise1 = nn.Conv1d(d_model, d_hidden, kernel_size=1)
        self.conv_pointwise2 = nn.Conv1d(d_hidden, d_model, kernel_size=1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv_depthwise(x.transpose(1, 2))
        x = self.dropout(self.act(self.conv_pointwise1(x)))
        return self.conv_pointwise2(x).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, d_model, d_hidden, dropout, ffn_dropout):
        super().__init__()
        # Keep module names aligned with the original research implementation.
        self.MHA = ConvAttention(d_model)
        self.feedforward = ConvFFN(d_model, d_hidden, ffn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layerNormal_1 = nn.LayerNorm(d_model)
        self.layerNormal_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layerNormal_1(x + self.dropout(self.MHA(x)))
        return self.layerNormal_2(x + self.dropout(self.feedforward(x)))


class LGTN(nn.Module):
    """Dual temporal/feature branches, global mean pooling and adaptive gating.

    Default values match the full L-GTN configuration in the experiments.
    The original configuration disables positional encoding and attention masks.
    The output contains seven unnormalized classification scores.
    """

    def __init__(self, time_steps=128, channels=5, num_classes=7,
                 d_model=64, d_hidden=128, depth=2, dropout=0.2,
                 ffn_dropout=0.1):
        super().__init__()
        if min(time_steps, channels, num_classes, d_model, d_hidden, depth) < 1:
            raise ValueError("Model dimensions and depth must be positive.")
        self.config = dict(time_steps=time_steps, channels=channels,
                           num_classes=num_classes, d_model=d_model,
                           d_hidden=d_hidden, depth=depth, dropout=dropout,
                           ffn_dropout=ffn_dropout)
        self.encoder_list_1 = nn.ModuleList([
            Encoder(d_model, d_hidden, dropout, ffn_dropout) for _ in range(depth)
        ])
        self.encoder_list_2 = nn.ModuleList([
            Encoder(d_model, d_hidden, dropout, ffn_dropout) for _ in range(depth)
        ])
        self.embedding_channel = nn.Linear(channels, d_model)
        self.embedding_input = nn.Linear(time_steps, d_model)
        self.gate = nn.Linear(2 * d_model, 2)
        self.output_linear = nn.Linear(2 * d_model, num_classes)

    def forward(self, x):
        temporal = self.embedding_channel(x)
        features = self.embedding_input(x.transpose(-1, -2))
        for encoder in self.encoder_list_1:
            temporal = encoder(temporal)
        for encoder in self.encoder_list_2:
            features = encoder(features)
        temporal = temporal.mean(dim=1)
        features = features.mean(dim=1)
        gates = F.softmax(self.gate(torch.cat((temporal, features), dim=-1)), dim=-1)
        combined = torch.cat((temporal * gates[:, 0:1], features * gates[:, 1:2]), dim=-1)
        return self.output_linear(combined)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
