import numpy as np
import time
    
def edit_dist(x, y):
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

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
    