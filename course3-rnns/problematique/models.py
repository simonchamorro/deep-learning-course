# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        self.encoder = nn.GRU(input_size=2, num_layers=n_layers, \
                           hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.GRU(input_size=hidden_dim, num_layers=n_layers, \
                           hidden_size=hidden_dim, batch_first=True)

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        # À compléter

    def forward(self, x):
        import pdb; pdb.set_trace()
        return None
    

