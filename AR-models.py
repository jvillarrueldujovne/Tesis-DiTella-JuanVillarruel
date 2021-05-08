# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:42:09 2021

@author: Juan
"""

"""
Este archivo contiene los estudios hechos sobre los modelos autorregresivos.
"""

# %% Importación de librerías y dependencias.

import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import yule_walker
import yfinance as yf
import matplotlib.patheffects as pe
# create and evaluate an updated autoregressive model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller






# %% Obtención de datos.

tickers = ['SPY'] # Lista de Tickers.

df = [] # En este dataframe se almacenarán todas las series temporales de los datos.
contador = 0

for ticker in tickers:
    print(ticker)
    data = yf.download(ticker, period='10y') # Todos los precios diarios de 10 años.
    if contador == 0:
        df = cp.deepcopy(data)
        df.drop(columns=['Open','High','Low','Adj Close','Volume'],inplace = True)
        df.rename(columns = {'Close':ticker}, inplace = True)
    else:
        df[ticker] = data['Close']
        df.rename(columns = {'Close':ticker}, inplace = True)
    contador += 1




# %% Test de Dickey-Fuller aumentado para evaluar la estacionalidad de las series.

df_test = df.rename_axis('ID').values # Convierto a un ndarray
shape = df_test.shape # Forma de mis datos
# series = df['SPY'].rename_axis('ID').values # SPY
series = df_test[:,0] # SPY Returns
X = series
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# %% Los retornos son creados por la función pct_change.


df2 = df.pct_change(periods=1, fill_method='pad', limit=None, freq=None)
df3 = df2.iloc[1:] # Se elimina la primer fila para eliminar los nan.


# %% Cambio a un ndarray
df_all = df3.rename_axis('ID').values # Convierto a un ndarray
shape = df_all.shape # Forma de mis datos


# %% Separación entre sets de entrenamiento y evaluación.

X = df_all
pct_train = 0.8
train, test = X[0:round(len(X)*pct_train)], X[round(len(X)*pct_train):]
ini = len(train)
fechas = df3.index.values[ini:(df.shape[0]-1)]

# %% Modelo AR(1)


rho, sigma = yule_walker(train, 1, method='mle') # Obtención de los parámetros del modelo.
rho_og, sigma_og = cp.deepcopy(rho), cp.deepcopy(sigma)
order = 1

y_1_sin =  list()
test_start = ini
t=0
value = test[0]
train2 = cp.deepcopy(train)
for i in range(len(test)): # Ciclo de predicción.
    window = X[ini-order+i:ini+i] #
    yhat = 0
    yhat = (rho_og * window + sigma_og)  # Cálculo del próximo valor.
    y_1_sin.append(yhat)


# %% Modelo AR(48) con y sin reentrenamiento.

order = 48
rho_50, sigma_50 = yule_walker(train, order, method='mle') # Calculo de los parámetros del modelo.
rho_50_og, sigma_50_og = cp.deepcopy(rho_50), cp.deepcopy(sigma_50)

y_50_sin =  list()
y_50_con = list()
test_start = round(len(X)*pct_train)
t=0
train2 = cp.deepcopy(train)

for i in range(len(test)):
    window_50 = X[test_start-order+i:test_start+i] # Ciclo de predicción.

    yhat = 0
    for k in range(len(window_50)):
        yhat += rho_50_og[k] * window_50[-1-k]
    yhat += sigma_50_og
    y_50_sin.append(yhat)



# %% Pasamos los retornos predichos a porcentaje.
y1sin = y_1_sin - np.mean(y_1_sin)
y50sin = y_50_sin - np.mean(y_50_sin)

true_y1sin = list()
true_y50sin = list()
test_scaled = list()

aaa = list()

for i in range(len(y_1_sin)):

    true_y1sin.append(round(y1sin[i][0][0]*100,2))
    true_y50sin.append(round(y50sin[i][0]*100,2))
    test_scaled.append(round(test[i][0]*100,2))

    aaa.append(round(y1sin[i][0][0],2))



# %% Gráfico de retornos del modelo AR(1)
fec = df.index.values[ini:(df.shape[0])-1]

fsize = 20
plt.figure()
plt.title('AR(1)',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(fec,test_scaled,label = 'Valor Real',linewidth=2,path_effects=[pe.Stroke(linewidth=0, foreground='black'), pe.Normal()])
plt.xlabel('Muestra',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('Retorno SPY (%)',fontsize=fsize)
plt.yticks(fontsize=fsize)
# plt.plot(y_test,label = 'Valor Real')
plt.plot(fec,true_y1sin,label = 'Predicción',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
# plt.plot(true_y1con,label = 'Predicción Reentrenada')
plt.legend(fontsize = fsize)
plt.show()

# %% Gráfico de retornos del modelo AR(48)

fec = df.index.values[ini:(df.shape[0])-1]

fsize = 20
plt.figure()
plt.title('AR(48)',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(fec,test_scaled,label = 'Valor Real',linewidth=2,path_effects=[pe.Stroke(linewidth=0, foreground='black'), pe.Normal()])
plt.xlabel('Muestra',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('Retorno SPY (%)',fontsize=fsize)
plt.yticks(fontsize=fsize)
# plt.plot(y_test,label = 'Valor Real')
plt.plot(fec,true_y50sin,label = 'Predicción',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
# plt.plot(true_y1con,label = 'Predicción Reentrenada')
plt.legend(fontsize = fsize)
plt.show()


# %% Reconstrucción de Precios diarios


ini = len(train)
ini_og = ini
precios_sin = list()
precios_con = list()
for i in range(len(test)):
    precios_sin.append(df.iloc[ini+i,0]*(1+true_y1sin[i]/100)) # Reconstrucción.


true_vals_precio = df.iloc[ini:(df.shape[0]-1),0].values
fechas = df.index.values[ini:(df.shape[0]-1)]


fsize = 20
plt.figure()
plt.title('Precio de SPY vs Predicción : Modelo AR(1)',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(fechas[:len(precios_sin)],true_vals_precio[:len(precios_sin)],label = 'Valor Real',linewidth=1,path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
plt.xlabel('Fecha',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('SPY (USD)',fontsize=fsize)
plt.yticks(fontsize=fsize)

# plt.plot(precios_sin,label = 'Predicción')
plt.plot(fechas[:len(precios_sin)],precios_sin,label = 'Predicción',linewidth=1,path_effects=[pe.Stroke(linewidth=1, foreground='red'), pe.Normal()]) # Reentrenada
# plt.plot(rec2,label = 'Predicción Filtrada')
plt.legend(fontsize = fsize)
plt.show()

# %% Reconstrucción de Precios semanales

ini = len(train)
ini_og = ini
precios_sin_50 = list()

for i in range(len(test)):
    precios_sin_50.append(df.iloc[ini+i,0]*(1+true_y50sin[i]/100))



true_vals_precio = df.iloc[ini:(df.shape[0]-1),0].values
fechas = df.index.values[ini:(df.shape[0]-1)]


fsize = 20
plt.figure()
plt.title('Precio de SPY vs Predicción : Modelo AR(48)',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(fechas[:len(precios_sin_50)],true_vals_precio[:len(precios_sin_50)],label = 'Valor Real',linewidth=1,path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
plt.xlabel('Fecha',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('SPY (USD)',fontsize=fsize)
plt.yticks(fontsize=fsize)

# plt.plot(precios_sin,label = 'Predicción')
plt.plot(fechas[:len(precios_sin_50)],precios_sin_50,label = 'Predicción',linewidth=1,path_effects=[pe.Stroke(linewidth=1, foreground='red'), pe.Normal()]) # Reentrenada
# plt.plot(rec2,label = 'Predicción Filtrada')
plt.legend(fontsize = fsize)
plt.show()

# %% RMSES

RMSE_ret_AR1 = np.sqrt(mean_squared_error(true_y1sin,test_scaled))
RMSE_ret_AR50 = np.sqrt(mean_squared_error(true_y50sin,test_scaled))

RMSE_price_AR1 = np.sqrt(mean_squared_error(true_vals_precio,precios_sin))
RMSE_price_AR50 = np.sqrt(mean_squared_error(true_vals_precio,precios_sin_50))

print(RMSE_ret_AR1,RMSE_ret_AR50)
print(RMSE_price_AR1,RMSE_price_AR50)