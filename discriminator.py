import torch
import torch.autograd as autograd
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, min_seq_len, max_seq_len, device, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.hidden2out = nn.Linear(hidden_dim * 2, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))
        return h.to(self.device)

    def forward(self, input, hidden, data_lens):  
        emb = self.embeddings(input)             
        emb = emb.permute(1, 0, 2)              
        output, hidden = self.gru(emb, hidden) 
        output = output.permute(1, 0, 2)       
        last_out = output[torch.arange(len(data_lens)).long(), (data_lens-1).long()]
        out = self.hidden2out(last_out)
        out = torch.sigmoid(out) 
        return out

    def batchClassify(self, inp, inp_len): 
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h, inp_len)
        return out.view(-1)

    def batchBCELoss(self, inp, target, inp_len):
        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h, inp_len)
        return loss_fn(out, target)

