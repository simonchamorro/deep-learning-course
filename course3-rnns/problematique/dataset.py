import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        # Create dictionnary to encode letters
        self.symb2int = {'<sos>': 0,
                         '<eos>': 1,
                         '<pad>': 2,
                         'a': 3,
                         'b': 4,
                         'c': 5,
                         'd': 6,
                         'e': 7,
                         'f': 8,
                         'g': 9,
                         'h': 10,
                         'i': 11,
                         'j': 12,
                         'k': 13,
                         'l': 14,
                         'm': 15,
                         'n': 16,
                         'o': 17,
                         'p': 18,
                         'q': 19,
                         'r': 20,
                         's': 21,
                         't': 22,
                         'u': 23,
                         'v': 24,
                         'w': 25,
                         'x': 26,
                         'y': 27,
                         'z': 28}

        self.int2symb = {}
        for k, v in self.symb2int.items():
            self.int2symb[v] = k

        self.dict_size = len(self.int2symb.keys())
        self.max_len = 0
        for i in range(len(self.data)):
            if len(self.data[i][1][0]) > self.max_len:
                self.max_len = len(self.data[i][1][0])

        # Extraction des symboles
        self.labels = [[l for l in data[0]] for data in self.data]
        self.one_hot_label = []

        # Ajout du padding aux labels
        for label in self.labels:
            label.append(stop_symbol)
            label.extend([pad_symbol]*(6-len(label)))
            self.one_hot_label.append([self.symb2int[l] for l in label])
        
        # Ajout du padding aux données
        self.input_len = max([d[1].shape[1] for d in self.data])
        x_len = np.mean([d[1][0][-1]/len(d[1][0]) for d in self.data])
        for data in self.data:
            last_x_val = data[1][0,-1]
            last_y_val = data[1][1,-1]
            pad = np.full((data[1].shape[0], self.input_len-data[1].shape[1]), np.expand_dims(data[1][:,-1], axis=1))
            # Pad as line 
            # for i in range(len(pad[0])):
            #     pad[0][i] = last_x_val + i*x_len
            data[1] = np.concatenate((data[1], pad), axis=1)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords = self.data[idx][1].T.copy()
        coords[:,0] *= 1.0/coords[:,0].max()
        coords[:,1] *= 1.0/coords[:,1].max()
        return torch.tensor(coords, dtype=torch.float32), \
               torch.tensor(self.one_hot_label[idx]).long()

    def visualisation(self, idx):
        # Get item
        sample, label = self[idx]

        # Visualisation des échantillons
        fig = plt.figure()
        plt.plot(sample[:,0], sample[:,1])
        plt.axis('equal')
        plt.legend([self.data[idx][0]])
        plt.title('Sample ' + str(idx), fontsize=10)
        


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(3):
        a.visualisation(np.random.randint(0, len(a)))
    plt.show()