# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Definition des couches
        # Couches pour rnn
        self.embedding = nn.Embedding(self.dict_size, hidden_dim)
        self.encoder = nn.GRU(input_size=2, num_layers=n_layers, \
                           hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.GRU(input_size=hidden_dim, num_layers=n_layers, \
                           hidden_size=hidden_dim, batch_first=True)

        # Couches pour attention et sortie
        self.att_combine = nn.Linear(2*hidden_dim, self.dict_size)
        self.hidden2query = nn.Linear(hidden_dim, hidden_dim)

        # Couche dense pour la sortie
        self.to(device)

    def forward(self, x):
        # Encoder
        out_enc, hidden = self.encoder(x)
        
        # Decoder
        batch_size = hidden.shape[1]

        # Init variables
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() 
        vec_out = torch.zeros((batch_size, self.max_len['output'], self.dict_size)).to(self.device) 
        attn_ws = torch.zeros((batch_size, self.max_len['input'], self.max_len['output'])).to(self.device) 

        # Output loop
        for i in range(self.max_len['output']):
            vec_in = self.embedding(vec_in)
            out, hidden = self.decoder(vec_in, hidden)
            attn_out, attn_w = self.attention(out, out_enc)
            out = torch.cat((out, attn_out.unsqueeze(1)), dim=1)
            out = self.att_combine(out.view(out.shape[0], -1))[:,None,:]
            vec_out[:,i:i+1,:] = out
            vec_in = torch.argmax(out, dim=2)
            attn_ws[:,:,i] = attn_w.squeeze(-1)

        return vec_out, hidden, attn_ws

    def attention(self, query, values):
        # Fully connected
        query = self.hidden2query(query)

        # Attention
        att = torch.bmm(query, values.permute(0,2,1))
        attn_w = torch.softmax(att[:,0,:], dim=1).unsqueeze(-1)
        attn_out = torch.sum(attn_w.repeat(1, 1, self.hidden_dim) * values, dim=1)
        return attn_out, attn_w
        

