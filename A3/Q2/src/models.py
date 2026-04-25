import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class ResidualSAGENet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes,
        num_layers=3,
        dropout=0.5,
    ):
        super().__init__()
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.input_bn = BatchNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        self.out_lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.input_lin(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        return self.out_lin(x)

class ResidualLinkSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_bn = BatchNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def encode(self, x, edge_index):
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x_skip = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        x = x + x_skip
        x = F.normalize(x, p=2, dim=1)
        return x

    def decode(self, z, edge_pairs):
        z_src = z[edge_pairs[:, 0]]
        z_dst = z[edge_pairs[:, 1]]
        prod = z_src * z_dst
        diff = (z_src - z_dst).abs()
        h = torch.cat([prod, diff], dim=1)
        return self.decoder(h).squeeze(-1)

    def forward(self, x, edge_index, edge_pairs=None):
        z = self.encode(x, edge_index)
        if edge_pairs is None:
            return z
        return self.decode(z, edge_pairs)
