# GRO722 Laboratoire 2
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Seq2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        # Encodeur
        embed = self.fr_embedding(x)
        out, hidden = self.encoder_layer(embed)       

        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            vec_in = self.en_embedding(vec_in)
            out, hidden = self.decoder_layer(vec_in, hidden)
            vec_out[:,i:i+1,:] = self.fc(out)
            vec_in = torch.argmax(out.clone().detach(), dim=2)
            
        return vec_out, hidden, None

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn


class Seq2seq_attn(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq_attn, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2*n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        # Encodeur
        embed = self.fr_embedding(x)
        out, hidden = self.encoder_layer(embed)       

        return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)

        # Attention
        att = torch.bmm(query, values.permute(0,2,1))
        attn_w = torch.softmax(att[:,0,:], dim=1).unsqueeze(-1)
        attn_out = torch.sum(attn_w.repeat(1, 1, self.n_hidden) * values, dim=1)

        return attn_out, attn_w

    def decoderWithAttn(self, encoder_outs, hidden):
        # Décodeur avec attention

        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage
        attention_weights = torch.zeros((batch_size, self.max_len['fr'], self.max_len['en'])).to(self.device) # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            vec_in = self.en_embedding(vec_in)
            out, hidden = self.decoder_layer(vec_in, hidden)
            attn_out, attn_w = self.attentionModule(out, encoder_outs)
            out = torch.cat((out, attn_out.unsqueeze(1)), dim=1)
            out = self.att_combine(out.view(out.shape[0], -1))[:,None,:]
            vec_out[:,i:i+1,:] = self.fc(out)
            vec_in = torch.argmax(out.clone().detach(), dim=2)
            attention_weights[:,:,i] = attn_w.squeeze(-1)

        return vec_out, hidden, attention_weights


    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out,h)
        return out, hidden, attn
