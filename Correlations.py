# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:23:09 2021

Este código busca analizar las correlaciones entre los activos
y la autocorrelación del ETF SPY.
"""

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

# import seaborn as sns
import copy as cp
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal as signal
import scipy.fftpack
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pywt
import time
import yfinance as yf



from matplotlib import pyplot
import statsmodels
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

# %% Obtención de datos.

"""
El primer paso es descargar todos los datos mediante la librería yfinance.
Se le especifica el ticker y el plazo del que se quiere obtener los datos.



SPY : ETF que sigue el valor del índice S&P-500.
GLD : ETF relacionado al Oro.
QQQ : ETF que sigue el precio del índice NASDAQ-100, de empresas tecnológicas.
VWO : ETF del rendimiento de acciones de mercado emergentes.
XLV : ETF del sector de Health.
VNQ : ETF del sector de Real Estate
XLE : ETF del sector enerégtico
XLP : ETF de productos básicos para consumo
XLF : ETF del sector Financiero
XLU : ETF de servicios básicos.
^TNX : Tasa del bono a 10 años de los Estados Unidos
^VIX : Índice de Volatilidad

"""


tickers = ['SPY','GLD','QQQ','VWO','XLV','VNQ','XLE','XLP','XLF','XLU',"^TNX","^VIX"] # Lista de Tickers.

df = [] # En este dataframe se almacenarán todas las series temporales de los datos.
contador = 0

for ticker in tickers:
    print(ticker)
    data = yf.download(ticker, period='10y',interval = "1wk")
    # data = yf.download(ticker, period='10y')
    if contador == 0:
        df = cp.deepcopy(data)
        df.drop(columns=['Open','High','Low','Adj Close','Volume'],inplace = True)
        df.rename(columns = {'Close':ticker}, inplace = True)
    else:
        df[ticker] = data['Close']
        df.rename(columns = {'Close':ticker}, inplace = True)
    contador += 1

# %% La tasa de 10 años (^TNX) tenía ciertos valores faltantes 'nan'.
# Esta sección los rellena con el valor anterior.
# Además, conviertes los datos finales a un 'ndarray' para un procesamiento más ágil.

a = df['^TNX'].isnull()
df3 = cp.deepcopy(df)

vals = list()
i=0
for values in df.index.values:
    if np.isnan(df.loc[values]['^TNX']):
        # print(df.loc[values]['^TNX'])
        df.loc[values,'^TNX'] =  df.loc[df3.index.values[i-1],'^TNX']
    i+=1

# %% Creo los pct_change para crear los retornos directo en el dataframe.

df2 = df.pct_change(periods=1, fill_method='pad', limit=None, freq=None)
df3 = df2.iloc[1:]

# %% Cambio a un ndarray
df_all = df3.rename_axis('ID').values # Convierto a un ndarray
shape = df_all.shape # Forma de mis datos
# %% Calculo de correlaciones

correlations = df3.corr(method='pearson')
correlations = correlations.round(3)

# %% Cálculo de autocorrelación.
nlag = 100
# autocor = pacf(df['SPY'], nlags=nlag, method='ywmle', alpha=None)
# plot_pacf(df['SPY'], lags=nlag)
autocor = pacf(df3.iloc[:,0]*100, nlags=nlag, method='ywmle', alpha=None)
plot_pacf(df3.iloc[:,0], lags=nlag)
pyplot.plot(autocor)
pyplot.xticks(fontsize=18)
pyplot.yticks(fontsize=18)
pyplot.xlabel('Lag',fontsize=20)
# pyplot.ylabel('PACF',fontsize=20)
pyplot.title('Autocorrelación parcial de los retornos de "SPY"',fontsize = 28)
pyplot.show()

# %% Test de Dickey-Fuller Aumentado

# series = df['SPY'].rename_axis('ID').values # SPY
series = df_all[:,0] # SPY Returns
X = series
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))



