import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import pandas as pd
import dgl
from dgl.nn import GraphConv
from weighted_gatconv import WeightedGATConv
import numpy as np

class GraphDataset(object):
    def __init__(self, datapath, name, fea_dim):
        super(GraphDataset, self).__init__()
        self.datapath = datapath
        self.name = name
        self.fea_dim = fea_dim

    def process(self):
        print(self.datapath, self.name)
        nodes_data = pd.read_csv(f'{self.datapath}/fea_dim_{self.fea_dim}_{self.name}_nodes.csv')
        edges_data = pd.read_csv(f'{self.datapath}/fea_dim_{self.fea_dim}_{self.name}_edges.csv')

        node_features = torch.from_numpy(nodes_data[nodes_data.columns[1:]].to_numpy())
        edge_features = torch.from_numpy(edges_data['weight'].to_numpy())
  
        edges_src = torch.from_numpy(edges_data['src'].to_numpy().astype(np.int32))
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy().astype(np.int32))
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])        
        self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['feat'] = node_features.float()
        self.graph.edata['weight'] = torch.concat([edge_features, torch.ones(nodes_data.shape[0])]).float()
        return self.graph

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layer_num, head_num, edge_type):
        super().__init__()

        if layer_num == 0:
            self.gat_linear = nn.Linear(in_size, out_size)
        elif layer_num == 1:
            self.gat_layers = nn.ModuleList(
                [dglnn.GATConv(in_size, out_size, head_num, activation=F.elu)]
            )
            self.gat_linear = nn.Linear(out_size*head_num, out_size)
        else:
            self.gat_layers = nn.ModuleList()
            self.gat_layers.append(dglnn.GATConv(in_size, hid_size, head_num, activation=F.elu))
            for _ in range(layer_num - 2):
                self.gat_layers.append(dglnn.GATConv(hid_size*head_num, hid_size, head_num, activation=F.elu))
            self.gat_layers.append(dglnn.GATConv(hid_size*head_num, out_size, 1, activation=F.elu))
            self.gat_linear = nn.Identity()


    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers) - 1: 
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layer_num):
        super().__init__()
        self.layers = nn.ModuleList()
        if layer_num == 1:
            self.layers.append(
                dglnn.GraphConv(in_size, out_size)
            )
        else:
            self.layers.append(
                dglnn.GraphConv(in_size, hid_size, activation=F.relu)
            )
            for i in range(layer_num - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size))
            self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layer_num):
        super().__init__()
        self.layers = nn.ModuleList()
        if layer_num == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, "gcn"))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
            for i in range(layer_num - 2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
class SoftmaxAttention(nn.Module):
    def __init__(self, feat_dim: int, num: int) -> None:
        super(SoftmaxAttention, self).__init__()
        self.trans = nn.Linear(feat_dim * num, feat_dim * num, bias=False)
        self.num = num
        self.query = nn.Linear(feat_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, embeds: list) -> torch.Tensor:
        num = len(embeds)
        batch, dim = embeds[0].shape
        x = torch.stack(embeds, dim=1)
        trans_x = self.trans(x.view(batch, num * dim)).tanh()
        weights = self.query(trans_x.view(batch, num, dim))
        weights = torch.softmax(weights.view(batch, num), dim=1)
        ans = torch.bmm(weights.unsqueeze(1), x)
        ans = self.layer_norm(ans.sum(dim=1))
        return ans, weights