import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy as sp
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter





#values = carga_csv('c:/Users/Daniel/Desktop/AprendizajeAutomatico/AprendizajeAutomatico/P1/ex1data2.csv')

#X = values[:, :-1]
#Y = values[:, -1]
m = np.shape(X)[0]
n = np.shape(X)[1]

Y = np.reshape(Y, (m, 1))
X = np.hstack([np.ones([m, 1]), X])




def normalize(mat):
    X_norm=np.empty(mat.shape)
    mu=np.empty(n+1)
    sigma=np.empty(n+1)

    for i in range(n+1):
        mu[i]=np.mean(mat[:,i]) #Media
        sigma[i]= np.std(mat[:,i]) #DesviacionStandar
        
        if sigma[i]!=0:
            aux=(X[:,i]-mu[i])/sigma[i]#cociente entre su diferencia con la media y la desviación estándar 
        else :
            aux=1

        X_norm[:,i]=aux

    return X_norm ,mu,sigma






def hipo(x, theta):
    return np.dot(x,np.transpose(theta))

#Thetas, costes = descenso_gradiente(X, Y, alpha)

def alg_grad(X, Y, thetas, alpha, j):
    sumatorio =  0.0
    for i in range(len(Y)):
        sumatorio = sumatorio+(hipo(X[i], thetas.T)- Y)*X[i][j]
    
    aux = (float(alpha)/len(Y))
    return thetas[j] -(aux*sumatorio)




#N -> num vars de X // nThetas = N+1 con 0, y 1 en la primera col extra 
#vector t = thetas.shape()
#for k in range(1500):
 #   for p in range(len(thetas)):
  #      t[p]= alg_grad(X, Y, thetas, 0.01, p)




