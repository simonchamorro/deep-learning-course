# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False          # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                   # Pour répétabilité
    n_workers = 0              # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    n_epochs = 50
    train_val_split = .7
    batch_size = 10 
    lr = 0.01
    n_hidden = 25
    n_layers = 1
    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')

    # Séparation du dataset (entraînement et validation)
    n_train_samp = int(len(dataset)*train_val_split)
    n_val_samp = len(dataset)-n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('\n')

    # Instanciation du model
    model = trajectory2seq(n_hidden, n_layers, dataset.int2symb, \
                           dataset.symb2int, dataset.dict_size, \
                           device, 5)
    model = model.to(device)

    # Afficher le résumé du model
    print('Model : \n', model, '\n')
    
    # Initialisation des variables
    best_val_loss = np.inf

    if trainning:

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            model.train()
            for batch_idx, data in enumerate(dataload_train):
                import pdb; pdb.set_trace()
                in_seq, target_seq = [obj.to(device).float() for obj in data]



                optimizer.zero_grad()
                pred = model(in_seq)
                loss = criterion(out, target_seq)
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()
            
                # Affichage pendant l'entraînement
                if batch_idx % 10 == 0:
                    print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                        epoch, n_epochs, batch_idx * len(data), len(dataload_train.dataset),
                                        100. * batch_idx / len(dataload_train), running_loss_train / (batch_idx + 1)), end='\r')
            
            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass