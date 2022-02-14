

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
   

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        
        self.embedding = nn.Embedding(output_size, emb_size)
        if model_type == 'RNN':
            self.recurrent = nn.RNN(emb_size, decoder_hidden_size, batch_first = True)
        elif model_type == "LSTM":
            self.recurrent = nn.LSTM(emb_size, decoder_hidden_size, batch_first = True)
            
        self.drop_out = nn.Dropout(dropout)
        self.ln = nn.Linear(decoder_hidden_size, output_size)
        self.log_sm = nn.LogSoftmax()
        

    def forward(self, input, hidden):
       
        #input =  torch.transpose(input, 0, 1)
        embedded = self.drop_out(self.embedding(input))
        
        
    
        output, hidden = self.recurrent(embedded, hidden)
       
       # print(output)
        output = self.log_sm(self.ln(output[:,0,:]))
        
        
       
        return output, hidden
