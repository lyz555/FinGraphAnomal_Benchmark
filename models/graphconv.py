# models/graphconv_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv 
class GCGraphConvModel(torch.nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, batchnorm=True):
        super(GCGraphConvModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2): 
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GraphConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)