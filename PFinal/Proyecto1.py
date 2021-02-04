import pandas as pd
from pandas import DataFrame
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import metrics
from sklearn import preprocessing


def rating_bynary_trans(rate):
    if rate < 3.5:
        return 0
    else:
        return 1



data = pd.read_csv('summer-products-with-rating-and-performance_2020-08.csv')



#data.info()
data.drop(labels = ["title","title_orig","currency_buyer", 'rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count', 'rating_one_count', 'badges_count', 'badge_local_product', 'badge_product_quality', 'badge_fast_shipping', 'tags', 'product_color', 'product_variation_size_id', 'product_variation_inventory', 'shipping_option_name', 'shipping_option_price', 'shipping_is_express', 'shipping_option_price', 'countries_shipped_to', 'inventory_total', 'has_urgency_banner', 'urgency_text', 'origin_country', 'merchant_title', 'merchant_name', 'merchant_info_subtitle', 'merchant_rating_count', 'merchant_rating', 'merchant_id', 'merchant_has_profile_picture', 'merchant_profile_picture', 'product_url', 'product_picture', 'product_id', 'theme', 'crawl_month'], axis = 1, inplace = True)

data["NiceRating"] = data["rating"].map(rating_bynary_trans)
data = data.reset_index(drop=True)

models_times = np.arange(4, dtype = float)


i=0
def graficaComparativa(campo,titulo,ticklabels):
    global i
    shape_cap = data[campo].value_counts()
    shape_labels = shape_cap.axes[0].tolist()
    inde = np.arange(len(shape_labels))

    lowRatedShape = []
    highRatedShape = []

    for shape in shape_labels:
        quantity = len(data[data[campo] == shape].index)
        highRated = len(data[(data[campo] == shape) & (data['NiceRating'] == 1)].index)
        highRatedShape.append(highRated)
        lowRatedShape.append(quantity-highRated)

    ancho = 0.4
    fig, ax = plt.subplots(figsize=(15,7))
    ax.bar(inde, highRatedShape, ancho, color='green')
    ax.bar(inde+ancho, lowRatedShape, ancho, color='red')

    ax.set_xlabel(campo, fontsize = 15)
    ax.set_ylabel('Productos',fontsize=15)  
    ax.set_title(titulo,fontsize=15)
    ax.set_xticks(inde+ancho/2)
    ax.set_xticklabels(ticklabels, fontsize=10)
    ax.legend(['High Rated', 'Not High Rated'])
    i = i + 1
    plt.show()
   
#graficaComparativa('units_sold', 'Diferencia de Ventas',
#['0-9','10', '10-100', '100-1k', '1k-10k', '10k-100k',
#'100k-1M' ,'1M-10M' ,'10M-100M', '100M-1B'])

#graficaComparativa('units_sold', 'Diferencia de Ventas',data["units_sold"].unique())
#demas graficas extra

#Mayor número de muestras comestibles que venenosas, como se puede observar en la gráfica
#Representamos la cantidad de muestras de ejmplo que son high_rated o low_rated a nivel gloval

def comparacion():
    highRated = []
    lowRated = []

    for cl in data["NiceRating"]:
        if cl==0:
            lowRated.append(cl)
        else:
            highRated.append(cl)

    xBars = ['High_rated: ' + str(len(highRated)), 'Low_rated:  ' + str(len(lowRated))]        
    ancho = 0.8
    fig, ax = plt.subplots(figsize=(8,7))
    index = np.arange(len(xBars))
    plt.bar(index, [len(highRated), len(lowRated)], ancho, color='blue')
    plt.xlabel('High_rated or Low_rated', fontsize=15)
    plt.ylabel('Quantity', fontsize=15)
    plt.xticks(index, xBars, fontsize=12, rotation=30)


def tranforData(d):
    #Rellenando los vacios
    #d.Size.fillna(method = 'ffill', inplace = True)

    d["precio_i"] = d["price"].astype(int)
    d["vendidas_i"] =np.floor(np.log10(d["units_sold"])).astype(int)
    d["rating_i"] = d["rating"].astype(int)


    return d


data = tranforData(data)


dataRate = data['NiceRating']
dataRating = DataFrame(dataRate)

dataRating.columns = ['NiceRating']

data.drop(labels = ['precio_i','rating_i','vendidas_i','uses_ad_boosts','rating','rating_count','NiceRating'], axis = 1, inplace = True)



labelEncode = preprocessing.LabelEncoder()

YArr = labelEncode.fit_transform(dataRate.values.ravel())
#print((YArr))

#data.drop(labels = ['NiceRating','rating_i',], axis = 1, inplace = True)

dataFeat = DataFrame(data)

#XArr = pd.get_dummies(dataFeat).values
XArr = pd.get_dummies(dataFeat.astype(str)).values
#print(dataRate)
#print((XArr))



#Regresion loogistica
m = len(YArr)

#La función sigmoide es la función h, la hipótesis
def sigmoide(value):
    s = 1/(1+np.exp(-value))
    return s

#FUNCIÓN DE COSTE
def coste(O, X, Y):
    H = sigmoide(np.dot(X,O))
    logH = np.log(H)
    logHT = logH.T
    logAux = np.log((1- H))
    logAuxT = logAux.T
    YT = Y.T
    suma = (-1/m)* (np.dot(YT, logH) + np.dot((1-YT), logAux))
    return suma
    
#FUNCIÓN DE GRADIENTE
def gradiente(O, X, Y):
    return (X.T.dot((sigmoide(X.dot(O))) - Y))/m

#FUNCIÓN DE COSTE REGULARIZADA (lambda)
def coste2(O, X, Y, lam):
    sol = (coste(O, X, Y) + (lam/(2*m))*(O**2).sum())
    return sol
   
#FUNCIÓN DE GRADIENTE REGULARIZADA (lambda)
def gradiente2(O, X, Y, lam):
    AuxO = np.hstack([np.zeros([1]), O[1:,]])
    return (((X.T.dot(sigmoide(X.dot(O))-Y))/m) + (lam/m)*O)

X = XArr.copy()
X = np.insert(X, 0, 1, axis = 1)

start = time.time()
thetas = np.ones(len(X[0]))
result = opt.fmin_tnc(func = coste2, x0 = thetas, fprime = gradiente2, args = (X, YArr, 0.1))
thetas_opt = result[0]
end = time.time()
print("EXE TIME:", end - start, "seconds")
print("OPT THETAS:\n", thetas_opt)


#Evaluación de los resultados obtenidos en las predicciones con las thetas óptimas
def evalua(thetas, X, y):
    thetasMat = np.matrix(thetas)   
    z = np.dot(thetasMat,X.transpose())
    resultados = sigmoide(z)
    resultados[resultados >= 0.5] = 1
    resultados[resultados < 0.5] = 0
    admitidosPred = sum(np.where(resultados == y)).shape[0]
    return (admitidosPred / len(y)) * 100


prediction = evalua(thetas_opt, X, YArr)
models_times[0] = (prediction)
print(models_times)
print("PREDICTIONS RESULT:",prediction)



#data.info()
#print(data)



#Red Neuronales:

from scipy.optimize import minimize as sciMin
from scipy.io import loadmat


lambda_ = 1


#SVM
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import svm

XSvm = XArr.copy()
YSvm = YArr.copy()

X_Train, X_test, Y_Train, Y_test = train_test_split(XSvm, YSvm, test_size=0.33, random_state=42)

#SVM de tipo lineal
svmLineal = svm.SVC(C=1, kernel='linear')

#Entrenamiento de las "redes". Similar a buscar las thetas óptimas
start = time.time()
print("\n TRAINING STARTED")
svmFitted = svmLineal.fit(X_Train, Y_Train)
end = time.time()
print("\n TRAINING FINISHED")
print("\n TRAINING EXECUTION TIME:", end - start, "seconds")
#predecimos Y a partir de la x "entrenada"
predictY = svmLineal.predict(X_test)

def evalua(results, Y):
    numAciertos = 0
    for i in range(len(Y_test)):
        if results[i] == Y[i]: numAciertos += 1
    return (numAciertos/(len(Y_test)))*100

success = evalua(predictY, Y_test)
models_times[2] = success
print("\nPREDICTIONS SVM LINEAL",success)