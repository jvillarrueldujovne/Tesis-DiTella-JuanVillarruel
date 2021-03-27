# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:42:09 2021

@author: Juan
"""

""" 
Este archivo contiene los estudios hechos sobre los modelos autorregresivos.
"""


# import seaborn as sns
import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import yule_walker
import yfinance as yf

# create and evaluate an updated autoregressive model
from statsmodels.tsa.ar_model import AutoReg


# %% Descarga de datos.

"""
Descarga de datos de "SPY"
"""

tickers = ['SPY']

df = []
contador = 0

for ticker in tickers:
    print(ticker)
    data = yf.download(ticker, period='10y')
    if contador == 0:
        df = cp.deepcopy(data)
        df.drop(columns=['Open','High','Low','Adj Close','Volume'],inplace = True)
        df.rename(columns = {'Close':ticker}, inplace = True)
    else:
        df[ticker] = data['Close']
        df.rename(columns = {'Close':ticker}, inplace = True)
    contador += 1



# %% Secuencias : División entre datos de entrenamiento y evaluación.



series = df['SPY']
# split dataset
X = series.values
pct_train = 0.8
train, test = X[1:round(len(X)*pct_train)-1], X[round(len(X)*pct_train):]


# %% AR(1) con y sin reentrenamiento.


rho, sigma = yule_walker(train, 1, method='mle')
rho_og, sigma_og = cp.deepcopy(rho), cp.deepcopy(sigma)
# Pasamos a la evaluación con entrenamiento.

y_1_sin =  list()
test_start = round(len(X)*pct_train)
t=0
window_1 = X[test_start+t-1:test_start+t]
train2 = cp.deepcopy(train)
y_1_con = list()
for i in range(len(test)):
    # Con Reentrenamiento
    yhat_1 = 0
    yhat_1 = rho * window_1 + sigma
    y_1_con.append(yhat_1)

    # Sin Reentrenamiento
    yhat_2 = 0
    yhat_2 = rho_og * window_1 + sigma_og
    y_1_sin.append(yhat_2)

    train2 = np.append(train2,test[t])
    # train2.append(test(t))
    t+=1
    window_1 = X[test_start+t-1:test_start+t]
    rho, sigma = yule_walker(train2, 1, method='mle')
    print(window_1,yhat_1,yhat_2,len(train2),train2[-1])


# %% Modelo AR(50) con y sin reentrenamiento.

rho_50, sigma_50 = yule_walker(train, 50, method='mle')

window_50 = X[test_start+t-50:test_start+t]
rho_50_og, sigma_50_og = cp.deepcopy(rho_50), cp.deepcopy(sigma_50)
# Pasamos a la evaluación con entrenamiento.

y_50_sin =  list()
y_50_con = list()
test_start = round(len(X)*pct_train)
t=0
train2 = cp.deepcopy(train)

for i in range(len(test)):
    yhat = 0
    
    # Con Reentrenamiento
    for k in range(len(window_50)):
        # print(t)
        yhat += rho_50[k] * window_50[-1-k]
    yhat += sigma_50
    y_50_con.append(yhat)

    # Sin Reentrenamiento
    yhat = 0
    for k in range(len(window_50)):
        yhat += rho_50_og[k] * window_50[-1-k]
    yhat += sigma_50_og
    y_50_sin.append(yhat)

    train2 = np.append(train2,test[t])
    # train2.append(test(t))
    t+=1
    window_50 = X[test_start+t-50:test_start+t]
    rho_50, sigma_50 = yule_walker(train2, 50, method='mle')


# %% 

y_1_con.pop(0)
y_1_sin.pop(0)

# %%
fsize = 20

plt.title('AR(50)',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(test,label = 'Valor Real')
plt.xlabel('Muestra',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('SPY (USD)',fontsize=fsize)
plt.yticks(fontsize=fsize)
# plt.plot(y_test,label = 'Valor Real')
plt.plot(y_50_sin,label = 'Predicción')
plt.plot(y_50_con,label = 'Predicción Reentrenada')
plt.legend(fontsize = fsize)
plt.show()

# %% Cálculo de los errores cuadráticos medios.



RMSE_sin = mean_squared_error(y_1_sin, test[0:len(y_1_sin)])
RMSE_con = mean_squared_error(y_1_con, test[0:len(y_1_con)])

RMSE_1 = [RMSE_sin,RMSE_con]

RMSE_sin = mean_squared_error(y_50_sin, test[0:len(y_50_sin)])
RMSE_con = mean_squared_error(y_50_con, test[0:len(y_50_con)])

RMSE_50 = [RMSE_sin,RMSE_con]


# %%

rho_50_2 = cp.deepcopy(rho_50)
rho_50_2 = np.append([0],rho_50)

# %% Gráfico de coeficientes del proceso AR.

plt.figure()
plt.title('Coeficientes del proceso AR(50)',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.stem(rho_50_2,label = 'Parámetro Número (i)')
plt.xlabel('i',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize = fsize)
plt.show()