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
        letter2hot = {'<pad>': 0,
                      '<sos>': 1,
                      '<eos>': 2,
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

        # Extraction des symboles
        self.labels = [[l for l in data[0]] for data in self.data]
        self.one_hot_label = []

        # Ajout du padding aux séquences
        for label in self.labels:
            label.insert(0, start_symbol)
            label.append(stop_symbol)
            label.extend([pad_symbol]*(7-len(label)))
            self.one_hot_label.append([letter2hot[l] for l in label])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.one_hot_label[idx], self.data[idx][1]

    def visualisation(self, idx):
        # Get item
        label, sample = self[idx]

        # Visualisation des échantillons
        fig = plt.figure()
        plt.plot(sample[0], sample[1])
        plt.axis('equal')
        plt.legend([self.data[idx][0]])
        plt.title('Sample ' + str(idx), fontsize=10)
        


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(3):
        a.visualisation(np.random.randint(0, len(a)))
    plt.show()