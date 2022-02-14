import random



import torch
import torch.nn as nn
import torch.optim as optim





class Seq2Seq(nn.Module):
    

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.to(device)
        self.decoder.to(device)
        

    def forward(self, source, out_seq_len=None):
        

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        if out_seq_len is None:
            out_seq_len = seq_len
        
        

       
        input = source[:, 0].view(batch_size,1)
        outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(self.device)
        
        if self.encoder.model_type == "RNN":
            
            en_out, hidden = self.encoder(source)
            for i in range(out_seq_len):
            
                output, hidden  = self.decoder(input, hidden)          
                outputs[:,i,:] = output           
                input = output.argmax(1).view(batch_size,1)
            return outputs
        
        
        
        
        
        
        hidden = self.encoder(source)
        for i in range(out_seq_len):
            
            output, hidden  = self.decoder(input, hidden)
           
            outputs[:,i,:] = output
           
            input = output.argmax(1).view(batch_size,1)
       
        return outputs
