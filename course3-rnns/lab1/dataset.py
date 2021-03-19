# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle

class SignauxDataset(Dataset):
    """Ensemble de signaux continus à differentes fréquences"""
    def __init__(self, filename='data.p'):
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        self.data = torch.as_tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):        
        return self.data[idx][0], self.data[idx][1]
    
    def visualize(self, idx):
        input_sequence, target_sequence = [i.numpy() for i in self[idx]]
        t = range(len(input_sequence)+1)
        plt.plot(t[:-1],input_sequence, label='input sequence')
        plt.plot(t[1:],target_sequence,label='target sequence')
        plt.title('Visualization of sample: '+str(idx))
        plt.legend()
        plt.show()

        pass

if __name__=='__main__':
    a = SignauxDataset()
    dataload_train = DataLoader(a, batch_size=2, shuffle=True)
    a.visualize(0)