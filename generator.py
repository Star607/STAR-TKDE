import logging
import dgl
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.nn.init as init
from graph_model import GAT, GCN, SAGE, GraphDataset, SoftmaxAttention
import helpers

class GraphGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, min_seq_len, max_seq_len, device='', len_list=None, starting_dist=None, node_fea_dim = 32, propath=None, edge_type = 'adjacent', node_emb_init_type='default', graph_names=None, graph_channels=None, p_stay='learnable', gmodel='GAT', g_hid_dim=32, g_out_dim=32, head_num = 4, layer_num = 1, decay_rate = 1, oracle_init=False):
        super(GraphGenerator, self).__init__()
        self.prefeature = True
        self.edge_type = edge_type
        self.node_emb_init_type = node_emb_init_type
        self.g_l = []
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.device = device 
        self.starting_dist = torch.tensor(starting_dist)
        self.len_vals, self.len_cnts = np.unique(len_list, return_counts=True) 
        self.len_cnts = self.len_cnts / np.sum(self.len_cnts)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim) 

        if self.prefeature:
            self.features_l = nn.ParameterList()
            for name in graph_names:
                g = GraphDataset(propath, name, node_fea_dim).process()
                features = g.ndata["feat"]
                assert features.shape[1] == node_fea_dim  
                self.g_l.append(g.to(device))
                self.features_l.append(nn.Parameter(features, requires_grad=False))               
        
        try:
            if gmodel == 'GAT':
                self.gnn_channels = nn.ModuleList([GAT(node_fea_dim, g_hid_dim, g_out_dim, layer_num, head_num, edge_type) for _ in range(graph_channels)])
        except ValueError:
            print("Oops! Gmodel is error!")

        self.fusion_layer = SoftmaxAttention(g_out_dim, graph_channels)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim)
        self.gru2out = nn.Linear(self.hidden_dim, self.vocab_size)

        self.stay_linear = nn.Linear(self.hidden_dim, 1)

        if p_stay == 'learnable':
            self.learn_stay = True

        self.gamma = decay_rate

        if oracle_init:
            for p in self.parameters():
                init.normal_(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h.to(self.device)     
    
    def compute_gnn(self):
        gnn_emb = []
        if self.prefeature and self.node_emb_init_type == 'default':
            idx = torch.arange(self.vocab_size).to(self.device)
            features = self.embedding(idx)
            for gat, g in zip(self.gnn_channels, self.g_l):
                gnn_emb.append(gat(g, features))

        res = gnn_emb[0]
        for i in range(1, len(gnn_emb)):
            res += gnn_emb[i] 
        return res

    def forward(self, inp, hidden, gnn_emb=None, cur_seq = None): 
        pad_emb = torch.cat([gnn_emb, torch.zeros((1, gnn_emb.shape[1])).to(inp)])
        emb = pad_emb[inp]               
        emb = emb.view(1, -1, self.embedding_dim)              
        out, hidden = self.gru(emb, hidden)                     
        
        gru_out = self.gru2out(out.view(-1, self.hidden_dim))   
        next_poi_logits = F.log_softmax(gru_out, dim=1)        

        stay_prob = torch.full((inp.shape[0], 1), 0.99).to(out) 
        if cur_seq != None and len(cur_seq)!= 0:
            last_count = torch.sum(cur_seq == cur_seq[-1], axis = len(cur_seq.shape) - 1)
            stay_prob = F.sigmoid(self.stay_linear(out.view(-1, self.hidden_dim))) * torch.exp(-self.gamma * last_count.unsqueeze(-1))
        
        return stay_prob, next_poi_logits


    def sample(self, num_samples, mode = 'padding'): 
        samples_len = torch.IntTensor([torch.IntTensor(np.random.choice(self.len_vals, 1, p = self.len_cnts)) for i in range(num_samples)]).to(self.device)
        h = self.init_hidden(num_samples)
        idx = torch.LongTensor([torch.multinomial(self.starting_dist, 1) for i in range(num_samples)]).reshape(-1, 1).to(self.device)
        gen_seq_len = self.max_seq_len - 1
        get_seq_offset = 0
        start_t = 1

        gnn_emb = self.compute_gnn()
        idx_cond = idx.view(-1)  

        if mode == 'padding':            
            for t in range(gen_seq_len):
                stay, out = self.forward(idx_cond, h, gnn_emb = gnn_emb) 
                idx_next = torch.multinomial(torch.exp(out), 1)
                idx = torch.cat((idx, idx_next), dim=1)
                idx_cond = idx_next.view(-1)
            return idx[:, get_seq_offset:], samples_len
        
        else:
            with torch.no_grad():
                for i in range(gen_seq_len):
                    stay, out = self.forward(idx_cond, h, gnn_emb=gnn_emb, cur_seq=idx[:, get_seq_offset:]) 
                    out_next = torch.multinomial(torch.exp(out), 1).view(-1) 
                    if not self.learn_stay or (start_t == 0 and i == 0):
                        idx_next = out_next 
                    else: 
                        ids = torch.ones(num_samples, dtype=bool)
                        ids_stay = torch.where(stay>=0.5)[0] 
                        ids[ids_stay] = False

                        idx_next = torch.zeros(num_samples).type(torch.LongTensor).to(self.device)
                        idx_next[ids_stay] = idx[ids_stay, -1] 
                        idx_next[ids] = out_next[ids] 
              
                    idx_cond = idx_next
                    idx = torch.cat((idx, idx_next.reshape(-1, 1)), dim=1)

                pred = [idx[i][get_seq_offset:get_seq_offset+samples_len[i]].tolist() for i in range(len(idx))]
                return pred

    def batchNLLLoss(self, inp, target, target_len):
        weights = torch.ones_like(inp)
        weights = helpers.sequence_mask(weights, target_len)
        stay_loss = nn.BCELoss(reduce = False)
        multi_loss = nn.NLLLoss(reduce = False)

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)         
        h = self.init_hidden(batch_size)
        loss = 0
        l_s0, l_m0 = 0, 0
        out_list = []
        gnn_emb = self.compute_gnn()
        cur_seq = torch.zeros(batch_size, seq_len).type(torch.LongTensor).to(self.device)
        for i in range(seq_len): 
            stay, out = self.forward(inp[i], h, gnn_emb=gnn_emb, cur_seq=cur_seq[:, :i])
            if self.learn_stay:
                l_s = stay_loss(stay, (target[:, i]==inp[i]).unsqueeze(-1).to(stay)) 
                l_s0 += (l_s * weights[:, i].unsqueeze(-1)).sum() 
            
            out_list.append(out[torch.arange(batch_size), target[:, i]])

            l_m = multi_loss(out, target[:, i]) 
            l_m0 += (l_m * weights[:, i]).sum() 
            cur_seq[:, i] = out.argmax(axis = 1) 

        STAY_LOSS = l_s0 / (weights.sum().item() + 1e-7)
        MULTI_LOSS = l_m0 / (weights.sum().item() + 1e-7)       
        # loss = l_s0 + l_m0
        return STAY_LOSS + MULTI_LOSS

    def batchPGLoss(self, inp, target, reward, target_len):
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          
        target = target.permute(1, 0)    
        h = self.init_hidden(batch_size)
        mask_reward = torch.zeros((batch_size, seq_len)).to(target)
        for i in range(batch_size):
            mask_reward[i, :target_len[i]] = reward[i]

        loss = 0
        gnn_emb = self.compute_gnn()
        for i in range(seq_len):  
            stay, out = self.forward(inp[i], h, gnn_emb=gnn_emb) 
            loss += (-out[torch.arange(batch_size).to(target), target[i]]*mask_reward[:, i]).sum()
        return loss/target_len.sum()
