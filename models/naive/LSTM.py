"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.f_x_linear = nn.Linear(input_size, hidden_size)
        self.f_h_linear = nn.Linear(hidden_size, hidden_size)
        
        self.i_x_linear = nn.Linear(input_size, hidden_size)
        self.i_h_linear = nn.Linear(hidden_size, hidden_size)
        
        self.g_x_linear = nn.Linear(input_size, hidden_size)
        self.g_h_linear = nn.Linear(hidden_size, hidden_size)
        
        self.o_x_linear = nn.Linear(input_size, hidden_size)
        self.o_h_linear = nn.Linear(hidden_size, hidden_size)
        
       
        
        self.sig_layer = nn.Sigmoid()
        self.tanh_linear = nn.Tanh()
        
        self.w_x = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.b_x = nn.Parameter(torch.zeros(hidden_size))
        
        self.w_h = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate

        # f_t: the forget gate

        # g_t: the cell gate

        # o_t: the output gate

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()
        

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        T = x.size()[1]
       
        h_t = torch.zeros(x.shape[0], self.hidden_size)
        c_t = torch.zeros(x.shape[0], self.hidden_size)
        
        for t in range(T):
            
            
            
            f_x = x[:,t,:] @ self.w_x + self.b_x
            f_h = h_t @ self.w_h + self.b_h
            f_t = self.sig_layer(f_x + f_h)
            
            i_x = x[:,t,:] @ self.w_x + self.b_x
            i_h = h_t @ self.w_h + self.b_h
            i_t = self.sig_layer(i_x + i_h)
            
            g_x = x[:,t,:] @ self.w_x + self.b_x
            g_h =  h_t @ self.w_h + self.b_h
            g_t = self.tanh_linear(g_x + g_h)
            
            c_t = f_t * c_t + i_t * g_t
            
            o_x = x[:,t,:] @ self.w_x + self.b_x
            o_h =  h_t @ self.w_h + self.b_h
            o_t = self.sig_layer(o_x + o_h)
            h_t = o_t * self.tanh_linear(c_t) 
            

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
       
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
