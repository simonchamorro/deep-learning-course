# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(x, y):
    # Initialize matrix
    mat = np.zeros((len(x)+1, len(y)+1))

    for i in range(len(x)+1):
      mat[i,0] = i
    
    for j in range(len(y)+1):
      mat[0,j] = j

    # Calcul de la distance d'édition
    for i in range(len(x)):
      for j in range(len(y)):
        diag_val = mat[i,j]
        if not x[i] == y[j]:
          diag_val += 1 
        mat[i+1,j+1] = np.min([diag_val, 1+mat[i,j+1], 1+mat[i+1,j]])
    
    return mat[i+1,j+1]

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion
    confusion_m = np.zeros((29, 29))
    symb_count = np.zeros((29, 29))

    for i in range(len(true)):
        for j in range(len(true[0])):
            confusion_m[true[i][j], pred[i][j]] += 1
            symb_count[true[i][j], :] += 1

    confusion_m /= symb_count

    return confusion_m[3:,3:]
