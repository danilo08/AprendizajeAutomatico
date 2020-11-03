import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy as sp
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.optimize as opt


def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).to_numpy()
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


#values = carga_csv('c:/Users/Daniel/Desktop/AA/AprendizajeAutomatico/P2/ex2data1.csv')


def sigmoide(X):
    sig = 1 / (1 + np.exp(-X))
    return sig


def hipo(X, theta):
    h = sigmoide(np.dot(X, theta))
    return h

def coste(theta,X,Y):
    H = hipo(X,theta)
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost



def gradiente(theta,X,Y):
    H = hipo(X,theta)
    gradient= (np.matmul(np.transpose(X),H-Y))/len(Y) 
    return gradient

def calculaOPTtheta(theta):
    result = opt.fmin_tnc(func=coste , x0=theta , fprime=gradiente , args=(_X,_Y))
    return result[0]

def evaluacion(t,X,Y):
    p=0
    cont=0
    a=0
    total=len(Y)

    for i in X:
        if sigmoide(np.dot(i,t)) >=0.5:
            p=1
        else:
            p=0
        
        if Y[cont]== p:
            a+=1
        cont+=1
    
    p=a/total *100
    plt.text(30,25,str(p)+"% de aciertos")







values = carga_csv('ex2data1.csv')


_X = values[:, :-1]
_Y = values[:,-1]
np.shape(_X)
np.shape(_Y)
m = np.shape(_X)[0]
n = np.shape(_X)[1]




pos = np.where(_Y == 1)
plt.scatter(_X[pos ,  0],_X[pos ,  1],marker='+',c='k')

pos2 = np.where(_Y == 0)
plt.scatter(_X[pos2 ,  0],_X[pos2 ,  1],marker='*',c='y')


_X = np.hstack([np.ones([m, 1]), _X])






thetas=np.zeros(3)
aux=gradiente(thetas,_X,_Y)
print(aux)


theta_opt = calculaOPTtheta(thetas)

nX = values[:, :-1]

x1_min, x1_max = nX[:, 0].min(), nX[:, 0].max()
x2_min, x2_max = nX[:, 1].min(), nX[:, 1].max()

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
np.linspace(x2_min, x2_max))

h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)),
xx1.ravel(),
xx2.ravel()].dot(theta_opt))
h = h.reshape(xx1.shape)


plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='r')


evaluacion(theta_opt,_X,_Y)






plt.show()