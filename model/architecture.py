import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from model.configs import CONFIG

class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, part_num, features_num, outputs_size):
        super(Model, self).__init__()
        self.hidden_dim = 128
        self.batch_size = CONFIG["batch_size"]
        self.use_gpu = torch.cuda.is_available()
        self.part_num = part_num
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.word_embeddings.weight.data.uniform_(-1., 1.)
        self.num_layers = 1
        self.dropout = 0.3
        self.bilstm = nn.LSTM(emb_dim, self.hidden_dim // 2, batch_first=True, num_layers=self.num_layers,
                              dropout=self.dropout, bidirectional=True)

        self.LinearLayer = nn.Sequential(
            nn.Linear(self.hidden_dim + features_num, 256),
            nn.Linear(256, 128),
            nn.Linear(128, outputs_size)
        )
        self.hidden = self.init_hidden()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    # Use subsequence embeddings to get whole sequence embedding
    def subsequence_embedding(self, inputs): 
        whole_embed = []
        for part_idx in range(inputs.shape[1]):
            embed = self.word_embeddings(inputs[:, part_idx, :])  
            embed = torch.transpose(embed, dim0=2, dim1=1)  
            embed = self.avg_pool(embed) 
            whole_embed.append(embed)
        whole_embed = torch.cat(whole_embed, dim=2) 
        return whole_embed

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            hidden_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
            cell_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            hidden_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            cell_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return (hidden_state, cell_state)

    def attention(self, lstm_out, final_state):
        hidden = final_state.view(-1, self.hidden_dim, 1)
        attn_weights = torch.bmm(lstm_out, hidden).squeeze(2) 
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2) 
        return context

    def merge_embedding(self, inputs0, inputs1, inputs2):
        embed0 = self.subsequence_embedding(inputs0)
        embed1 = self.subsequence_embedding(inputs1)
        embed2 = self.subsequence_embedding(inputs2)
        return embed0, embed1, embed2

    # transform sequence embeddings of three forms into final embedding
    def forms2final(self, embed0, embed1, embed2):
        final_embed = []
        for i in range(embed0.shape[0]):
            form1 = embed0[i,:,:].unsqueeze(2)
            form2 = embed1[i,:,:].unsqueeze(2)
            form3 = embed2[i,:,:].unsqueeze(2)
            concat = torch.cat((form1,form2),axis=2)
            concat = torch.cat((concat,form3),axis=2)
            final = self.max_pool(concat)
            final_embed.append(final)
        final_embed = torch.cat(final_embed,dim=2)
        final_embed = torch.transpose(final_embed,0,2)
        final_embed = torch.transpose(final_embed,1,2)
        return final_embed

    def forward(self, inputs, features):
        inputs0, inputs1, inputs2 = inputs
        embed0, embed1, embed2 = self.merge_embedding(inputs0, inputs1, inputs2)
        embed = self.forms2final(embed0, embed1, embed2)
        embed = torch.transpose(embed, dim0=2, dim1=1) 
        hidden = self.init_hidden(embed.size()[0])
        lstm_out, hidden = self.bilstm(embed, hidden) 
        final_hidden_state, final_cell_state = hidden
        attn_out = self.attention(lstm_out, final_hidden_state) 
        outputs = torch.cat([attn_out, features], dim=1)     
        outputs = self.LinearLayer(outputs)
        return outputs