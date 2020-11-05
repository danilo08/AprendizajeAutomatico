import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy as sp
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.optimize as opt


import sklearn.preprocessing as sk



def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).to_numpy()
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


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


def coste2(theta,X,Y): 
    aux = (lamda/(2*m))
    sumatorio=0
    for i in range(n):
        sumatorio = sumatorio +theta[i]**2

    return coste(theta,X,Y) +aux * sumatorio



def gradiente(theta,X,Y):

    xt=np.transpose(X)
    g=sigmoide(np.matmul(X,theta))-Y

    aux=np.r_[[0],theta[1:]]
    
    grad=(1/m)* np.matmul(xt,g) + (lamda/m)*aux

    return grad

  
def calculaOPTtheta():
    result = opt.fmin_tnc(func=coste2 , x0=np.zeros(XPoly.shape[1]) , fprime=gradiente , args=(XPoly,_Y))
    return result[0]

        

  
    



values = carga_csv('ex2data2.csv')


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



poly =sk.PolynomialFeatures(6)
XPoly = poly.fit_transform(_X)

lamda=1

thetas=np.zeros(28)
print("Coste :",coste2(thetas,XPoly,_Y))


theta_opt = calculaOPTtheta()


x1_min, x1_max = XPoly[:, 0].min(), XPoly[:, 0].max()
x2_min, x2_max = XPoly[:, 1].min(), XPoly[:, 1].max()

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),np.linspace(x2_min, x2_max))
h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),xx2.ravel()]).dot(theta_opt))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='g')


plt.legend()
plt.show()