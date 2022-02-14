

import numpy as np

import torch
from torch import nn
import random


class TransformerTranslator(nn.Module):
   
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
       
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        self.emb = nn.Embedding(input_size, self.word_embedding_dim)
        self.pos_emb = nn.Embedding(max_length, self.word_embedding_dim)
        
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
       

        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)
        
        self.ff_1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.ff_2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.relu = nn.ReLU()
        self.norm_ff= nn.LayerNorm(self.hidden_dim)

        
        self.final_ln = nn.Linear(self.hidden_dim, self.output_size)
        self.sm = nn.Softmax(dim = 2)


        
    def forward(self, inputs):



        emb = self.embed(inputs)
        
        attention = self.multi_head_attention(emb)
        ff = self.feedforward_layer(attention)
        outputs = self.final_layer(ff)
        
       
        return outputs
    
    
    def embed(self, inputs):

        emb = self.emb(inputs)
        pos_en = torch.arange(0, inputs.shape[1]).unsqueeze(0).repeat(inputs.shape[0], 1)
        embeddings = emb + self.pos_emb(pos_en)
  
        return embeddings
        
    def multi_head_attention(self, inputs):

        dk = np.sqrt(self.dim_k)
        k_1 = self.k1(inputs)
        v_1 = self.v1(inputs)
        q_1 = self.q1(inputs)
        score_1 = self.softmax(torch.matmul(q_1, k_1.transpose(1,2)) / dk)
        head_1 = torch.matmul(score_1, v_1)

        
        

        k_2 = self.k2(inputs)
        v_2 = self.v2(inputs)
        q_2 = self.q2(inputs)
        score_2 = self.softmax(torch.bmm(q_2, k_2.transpose(1,2)) / dk)
        head_2 = torch.bmm(score_2, v_2)

        x = torch.cat((head_1, head_2), dim = 2)

        attention = self.attention_head_projection(x)
        outputs = self.norm_mh(inputs + attention)

        return outputs
    
    
    def feedforward_layer(self, inputs):
        

        ln_1 = self.ff_1(inputs)
        ffn = self.ff_2(self.relu(ln_1))
        outputs = self.norm_ff(inputs + ffn)

        return outputs
        
    
    def final_layer(self, inputs):
        
        

        outputs = self.final_ln(inputs)

        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
