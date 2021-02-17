import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Module du dataset
from voc_classification_dataset import VOCClassificationDataset

# Génération des "path"
dir_path = os.path.dirname(__file__)
data_path = dir_path + '/data'
images_path = dir_path + '/test_images'
weigths_path = dir_path + '/weights'

# ---------------- Parametres et hyperparamètres ----------------#
use_cpu = True  # Forcer à utiliser le cpu?
save_model = True  # Sauvegarder le meilleur modèle ?

input_channels = 3  # Nombre de canaux d'entree
num_classes = 21  # Nombre de classes
batch_size = 32  # Taille des lots pour l'entrainement
val_test_batch_size = 32  # Taille des lots pour validation et test
epochs = 2  # Nombre d'iterations (epochs)
train_val_split = 0.8  # Proportion d'échantillons
lr = 0.0001  # Taux d'apprentissage
random_seed = 1  # Pour repetabilite
num_workers = 6  # Nombre de threads pour chargement des données
input_size = 100  # Taille (l&h) des images desire
# ------------ Fin des parametres et hyperparamètres ------------#

if __name__ == '__main__':
    # Initialisation des objets et des variables
    best_val = np.inf
    torch.manual_seed(random_seed)
    np.random.seed(seed=random_seed)

    # Init device
    use_cuda = not use_cpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Affichage
    fig1, axs1 = plt.subplots(1)
    fig2, axs2 = plt.subplots(3, 2, dpi=100, figsize=(10, 10))

    # Chargement des datasets
    params_train = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': num_workers}

    params_val = {'batch_size': val_test_batch_size,
                  'shuffle': True,
                  'num_workers': num_workers}

    dataset_trainval = VOCClassificationDataset(data_path, image_set='train', download=True, img_shape=input_size)

    # Séparation du dataset (entraînement et validation)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_trainval,
                                                               [int(len(dataset_trainval) * train_val_split),
                                                                int(len(dataset_trainval) - int(
                                                                    len(dataset_trainval) * train_val_split))])

    print('Number of epochs : ', epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('\n')

    # Création des dataloaders
    train_loader = torch.utils.data.DataLoader(dataset_train, **params_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, **params_val)

    # ------------------------ Laboratoire 2 - Question 2 - Début de la section à compléter ----------------------------

    model = torchvision.models.resnet18(pretrained=True, progress=True)
    
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = nn.Sequential(model,
                          nn.ReLU(),
                          nn.Linear(model.fc.out_features, num_classes),
                          nn.Sigmoid())

    # ------------------------ Laboratoire 2 - Question 2 - Fin de la section à compléter ------------------------------

    model.to(device)
    print('Model : ')
    print(model)

    # Création de l'optmisateur et de la fonction de coût
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print('Starting training')
    epochs_train_losses = []  # historique des couts
    epochs_val_losses = []  # historique des couts

    for epoch in range(1, epochs + 1):
        # Entraînement
        model.train()
        running_loss = 0

        # Boucle pour chaque lot
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Affichage pendant l'entraînement
            if batch_idx % 10 == 0:
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), running_loss / (batch_idx + 1)), end='\r')

        # Historique des coûts
        epochs_train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        m_ap = 0
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # ------------------------ Laboratoire 2 - Question 3 - Début de la section à compléter ----------------
        true_pos_by_class_by_thresh_idx = [[0 for _ in thresholds] for _ in range(num_classes)]
        false_pos_by_class_by_thresh_idx = [[0 for _ in thresholds] for _ in range(num_classes)]
        false_neg_by_class_by_thresh_idx = [[0 for _ in thresholds] for _ in range(num_classes)]

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                target = target.detach().cpu().numpy()
                output = output.detach().cpu().numpy()

                for c in range(num_classes):
                    for thresh_idx in range(len(thresholds)):
                        pred = (output[:,c] > thresholds[thresh_idx]).astype(int)
                        target_c = target[:,c]
                        true_pos_by_class_by_thresh_idx[c][thresh_idx] += \
                            np.logical_and(pred == 1, target_c == 1).sum()
                        false_pos_by_class_by_thresh_idx[c][thresh_idx] += \
                            np.logical_and(pred == 1, target_c == 0).sum()
                        false_neg_by_class_by_thresh_idx[c][thresh_idx] += \
                            np.logical_and(pred == 0, target_c == 1).sum()

            for c in range(num_classes):
                recalls = []
                precisions = []
                for thresh_idx in range(len(thresholds)):
                    true_pos = true_pos_by_class_by_thresh_idx[c][thresh_idx]
                    false_pos = false_pos_by_class_by_thresh_idx[c][thresh_idx]
                    false_neg = false_neg_by_class_by_thresh_idx[c][thresh_idx]

                    recall_denom = true_pos + false_neg
                    recalls.append(true_pos / recall_denom if recall_denom > 0 else 0)

                    precision_denom = true_pos + false_pos
                    precisions.append(true_pos / precision_denom if precision_denom > 0 else 0)

                sorted_idx = np.argsort(recalls)
                sorted_precision = np.array(precisions)[sorted_idx]
                sorted_recall = np.array(recalls)[sorted_idx]

                m_ap += np.trapz(x=sorted_recall, y=sorted_precision)

            m_ap /= num_classes

        # ------------------------ Laboratoire 2 - Question 3 - Début de la section à compléter ----------------

        # Historique des coûts de validation
        val_loss /= len(val_loader)
        epochs_val_losses.append(val_loss)
        print('\nValidation - Average loss: {:.4f}, mAP: {:.4f}\n'.format(val_loss, m_ap))

        # Sauvegarde du meilleur modèle
        if val_loss < best_val and save_model:
            best_val = val_loss
            if save_model:
                print('\nSaving new best model\n')
                torch.save(model, dir_path + '/weights/no1_best.pt')

        # Affichage des prédictions
        for i in range(3):
            image = data[i].cpu()
            n_class = target.shape[0]
            pred = output

            axs2[i, 0].cla()
            axs2[i, 1].cla()
            axs2[i, 0].imshow(image.permute(1, 2, 0))
            axs2[i, 1].barh(range(num_classes), pred[i])
            axs2[i, 1].set_yticks(range(num_classes))
            axs2[i, 1].set_yticklabels(dataset_trainval.VOC_CLASSES_2_ID.keys())
            axs2[i, 1].set_xlim([0, 1])

        # Affichage des courbes d'apprentissage
        axs1.cla()
        axs1.set_xlabel('Epochs')
        axs1.set_ylabel('Loss')
        axs1.plot(range(1, len(epochs_train_losses) + 1), epochs_train_losses, color='blue', label='Training loss',
                  linestyle=':')
        axs1.plot(range(1, len(epochs_val_losses) + 1), epochs_val_losses, color='red', label='Validation loss',
                  linestyle='-.')
        axs1.legend()
        fig1.show()
        fig2.show()
        plt.pause(0.001)

    plt.show()
