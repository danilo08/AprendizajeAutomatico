import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import fmin_tnc as tnc
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pf
import sys


def rating_bynary_trans(rate):
    if rate < 4.0:
        return 0
    else:
        return 1



data = pd.read_csv('summer-products-with-rating-and-performance_2020-08.csv')
data = data.dropna()
data = data.reset_index(drop=True)
#data.info()
data.drop(labels = ["title","title_orig","currency_buyer", 'rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count', 'rating_one_count', 'badges_count', 'badge_local_product', 'badge_product_quality', 'badge_fast_shipping', 'tags', 'product_color', 'product_variation_size_id', 'product_variation_inventory', 'shipping_option_name', 'shipping_option_price', 'shipping_is_express', 'shipping_option_price', 'countries_shipped_to', 'inventory_total', 'has_urgency_banner', 'urgency_text', 'origin_country', 'merchant_title', 'merchant_name', 'merchant_info_subtitle', 'merchant_rating_count', 'merchant_rating', 'merchant_id', 'merchant_has_profile_picture', 'merchant_profile_picture', 'product_url', 'product_picture', 'product_id', 'theme', 'crawl_month'], axis = 1, inplace = True)

data["NiceRating"] = data["rating"].map(rating_bynary_trans)

models_times = np.arange(4, dtype = float)

data.head()#no se pork no va
data.info()



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
graficaComparativa('units_sold', 'Diferencia de Ventas',data["units_sold"])



