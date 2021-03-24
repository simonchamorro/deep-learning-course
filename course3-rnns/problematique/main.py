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
from tqdm import tqdm


def visualisation_test(sample, label, pred, attn):

    # Create plot
    fig, ax = plt.subplots(6) 
    fig.suptitle(letters2str(label) + '\n' + letters2str(pred))
    for axis in ax:
        axis.set_xlim([-1, 2])
        axis.set_ylim([-1, 2])
        axis.set_xticks([])
        axis.set_yticks([])
    ax[0].scatter(sample[:,0], sample[:,1], c=attn[0,:,5], s=1)
    plt.yticks(rotation=90)
    ax[1].scatter(sample[:,0], sample[:,1], c=attn[0,:,4], s=1)
    plt.yticks(rotation=90)
    ax[2].scatter(sample[:,0], sample[:,1], c=attn[0,:,3], s=1)
    plt.yticks(rotation=90)
    ax[3].scatter(sample[:,0], sample[:,1], c=attn[0,:,2], s=1)
    plt.yticks(rotation=90)
    ax[4].scatter(sample[:,0], sample[:,1], c=attn[0,:,1], s=1)
    plt.yticks(rotation=90)
    ax[5].scatter(sample[:,0], sample[:,1], c=attn[0,:,0], s=1)
    plt.yticks(rotation=90)
    plt.setp(ax[0], ylabel=pred[5])
    plt.setp(ax[1], ylabel=pred[4])
    plt.setp(ax[2], ylabel=pred[3])
    plt.setp(ax[3], ylabel=pred[2])
    plt.setp(ax[4], ylabel=pred[1])
    plt.setp(ax[5], ylabel=pred[0])
    plt.show()


def letters2str(letters):
    word = ""
    for l in letters:
        word += l
    return word



if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False          # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = False                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                   # Pour répétabilité
    n_workers = 0              # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    n_epochs = 30
    train_val_split = .7
    batch_size = 32
    lr = 0.01
    n_hidden = 25
    n_layers = 2
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
                           device, {'input': dataset.input_len, 'output': 6})
    model = model.to(device)

    # Afficher le résumé du model
    print('Model : \n', model)
    print('Params : \n', sum(p.numel() for p in model.parameters() if p.requires_grad), '\n')
    
    # Initialisation des variables
    best_val_loss = np.inf

    if trainning:

        # Plot curves
        if learning_curves:
            train_dist = [] 
            train_loss= [] 
            valid_dist = [] 
            valid_loss= [] 
            fig, ax = plt.subplots(1) 

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            
            # Entraînement
            dist_train = 0
            running_loss_train = 0
            model.train()
            for batch_idx, data in enumerate(dataload_train):

                # Format 
                in_seq, target_seq = data
                in_seq = in_seq.to(device)
                target_seq = target_seq.to(device).long()

                # Training
                optimizer.zero_grad()
                output, hidden, attn = model(in_seq)
                loss = criterion(output.view((-1, model.dict_size)), target_seq.view(-1))
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()
            
                # Edit distance
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(len(target_seq_list)):
                    a = target_seq_list[i]
                    b = output_list[i]
                    M = a.index(1)
                    dist_train += edit_distance(a[:M],b[:M])/batch_size
                
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist_train/len(dataload_train)), end='\r')

            
            # Validation
            dist_valid = 0
            running_loss_valid = 0
            model.eval()
            for batch_idx, data in enumerate(dataload_val):

                # Format 
                in_seq, target_seq = data
                in_seq = in_seq.to(device)
                target_seq = target_seq.to(device).long()

                # Training
                output, hidden, attn = model(in_seq)
                loss = criterion(output.view((-1, model.dict_size)), target_seq.view(-1))
                running_loss_valid += loss.item()
            
                # Edit distance
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(len(target_seq_list)):
                    a = target_seq_list[i]
                    b = output_list[i]
                    M = a.index(1)
                    dist_valid += edit_distance(a[:M],b[:M])/batch_size
            
            # Print epoch results
            print('\n')
            print('Valid - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx+1) * batch_size, len(dataload_val.dataset),
                    100. * (batch_idx+1) *  batch_size / len(dataload_val.dataset), running_loss_valid / (batch_idx + 1),
                    dist_valid/len(dataload_val)), end='\r')
            print('\n')
            print('\n')

            # Plot curves
            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                train_dist.append(dist_train/len(dataload_train))
                valid_loss.append(running_loss_valid/len(dataload_val))
                valid_dist.append(dist_valid/len(dataload_val))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.plot(valid_loss, label='validation loss')
                ax.plot(valid_dist, label='validation distance')
                ax.legend()
                plt.draw()
                plt.pause(0.01)
                plt.savefig('model_' + str(epoch) + '.png')

            # Save model
            torch.save(model,'model_' + str(epoch) + '.pt')

        # Plot learning curves after training
        if learning_curves:
            plt.show()
            plt.close('all')

    if test:
        # Test
        batch_size = 500
        dataset = HandwrittenWords('data_trainval.p')
        dataload = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        model = torch.load('model.pt')
        model.eval()

        dist = 0
        running_loss = 0
        pred = []
        gt = []
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        print('Testing on ' + str(len(dataset)) + ' samples...')
        for batch_idx, data in tqdm(enumerate(dataload)):

            # Format 
            in_seq, target_seq = data
            in_seq = in_seq.to(device)
            target_seq = target_seq.to(device).long()

            # Training
            output, hidden, attn = model(in_seq)
            loss = criterion(output.view((-1, model.dict_size)), target_seq.view(-1))
            running_loss += loss.item()
        
            # Edit distance
            output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
            target_seq_list = target_seq.cpu().tolist()
            pred.extend(output_list)
            gt.extend(target_seq_list)
            M = len(output_list[0])
            for i in range(len(target_seq_list)):
                a = target_seq_list[i]
                b = output_list[i]
                M = a.index(1)
                dist += edit_distance(a[:M],b[:M])/batch_size
        
        # Print results
        print('Results Test - Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                running_loss / (batch_idx + 1), dist/len(dataload)), end='\r')
        print('\n')

        # Affichage de la matrice de confusion
        letter_list = ['a','b','c','d','e',\
                       'f','g','h','i','j','k','l','m','n','o','p',\
                       'q','r','s','t','u','v','w','x','y','z']
        confusion_m = confusion_matrix(gt, pred)
        plt.figure()
        plt.imshow(confusion_m, origin='lower',  vmax=1, vmin=0, cmap='pink')
        plt.xticks(ticks=[i for i in range(26)], labels=letter_list, rotation=45)
        plt.yticks(ticks=[i for i in range(26)], labels=letter_list)
        plt.xlabel('pred')
        plt.ylabel('true')
        plt.show()

        # Visualize results
        for i in range(3):
            idx = np.random.randint(0, len(dataset))
            # Format 
            in_seq, target_seq = dataset[idx]
            in_seq = in_seq.to(device)
            target_seq = target_seq.to(device).long()

            # Training
            output, hidden, attn = model(in_seq.unsqueeze(0))
            output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()[0]
            target_seq_list = target_seq.cpu().tolist()
            label = [model.int2symb[l] for l in target_seq_list]
            pred = [model.int2symb[l] for l in output_list]

            visualisation_test(in_seq.detach().cpu(), label, pred, \
                               attn.detach().cpu())
                  



