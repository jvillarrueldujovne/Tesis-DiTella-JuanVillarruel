#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:57:31 2021
Finished on Sat Mar 27

@author: Juan Villarruel Dujovne

Esté codigo tiene todo el desarrollo hecho para el trabajo final de tesis
de la maestría en finanzas de la Universidad Di Tella.

El código está dividido en secciones.
"""

# %% Importación de librerías y dependencias.

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

# import seaborn as sns
import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import yfinance as yf


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

"""
El dataframe 'df' contendrá los valores originales de los precios de cierre.
"""

tickers = ['SPY','GLD','QQQ','VWO','XLV','VNQ','XLE','XLP','XLF','XLU',"^TNX","^VIX"] # Lista de Tickers.

df = [] # En este dataframe se almacenarán todas las series temporales de los datos.
contador = 0

for ticker in tickers:
    print(ticker)
    # data = yf.download(ticker, period='10y')
    data = yf.download(ticker, period='10y',interval = "1wk") # Obtención de los datos semanales
    if contador == 0:
        df = cp.deepcopy(data)
        df.drop(columns=['Open','High','Low','Adj Close','Volume'],inplace = True)
        df.rename(columns = {'Close':ticker}, inplace = True)
    else:
        try: # Por feriados y demases, ciertos valores figuran como 'nan'.
            df[ticker] = data['Close']
            df.rename(columns = {'Close':ticker}, inplace = True)
        except:
            df[ticker] = np.nan
            for date in df.index.values:
                if date in data.index.values:
                    try:
                        df.loc[date,ticker] = data.loc[date]['Close']
                    except:
                        df.loc[date,ticker] = np.nan

    contador += 1





# %% Data Padding : Muchos valores figurar como 'nan' y deben ser reconstruidos.
# Interpolamos con los valores previos y posteriores.

a = df['^TNX'].isnull()
df3 = cp.deepcopy(df)

vals = df.index.values
for ticker in tickers:
    i=0
    for values in df.index.values:
        if np.isnan(df.loc[values][ticker]):
            if np.isnan(df.loc[df3.index.values[i+1],ticker]):
                df.loc[values,ticker] =  (df.loc[df3.index.values[i-1],ticker] + df.loc[df3.index.values[i+2],ticker])/2
            else:
                df.loc[values,ticker] =  (df.loc[df3.index.values[i-1],ticker] + df.loc[df3.index.values[i+1],ticker])/2
        i+=1

# %% Aquí obtendremos los retornos de los distintos actios.
# La función pct_change permite obtener los retornos especificados.

df2 = df.pct_change(periods=1, fill_method='pad', limit=None, freq=None)
df3 = df2.iloc[1:]

# %% Cambio a un ndarray
df_all = df3.rename_axis('ID').values # Convierto a un ndarray
shape = df_all.shape # Forma de mis datos




# %% Escalamiento de los datos al rango (-1,1)

def ScaleData(scaler,data):
    # scaler = MinMaxScaler(feature_range=(-1,1))
    shape = data.shape
    data = data.reshape(shape[0]*shape[1],1)
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled = data_scaled.reshape(shape[0],shape[1])
    return data_scaled

scaler = MinMaxScaler(feature_range=(-1,1)) # Escalador entre (-1,1)
data_scaled = ScaleData(scaler,df_all) # Variable que contiene los datos originales reescalados.


# %% Separación de datos en Train - Test

def SecuenciasTrainingTest(datos,n_steps,pc_train = 0.9,lag=1):
    # datos es un DataFrame con los datos.
    # n_steps es la cantidad de días usados para predecir.
    # dias_delay es la cantidad de días para adelante que quiero predecir
    X , y = list(), list()
    len_datos = datos.shape[0]-lag-1
    for i in range(round(len_datos*pc_train)):
        seq_x, seq_y = list(), list()
        end_ix = i + n_steps
        end_l = i+n_steps+lag
        if end_l >= len_datos:
            break
        seq_x = datos[i:end_ix, : ]
        seq_y = datos[end_l][0] # La salida es la primer columna
        X.append(seq_x)
        y.append(seq_y)


    X_test, y_test = list(),list()
    # print("SWITCH")
    for i in range(round(len_datos*pc_train),len_datos):

        # print(i)
        seq_x2, seq_y2 = list(), list()
        end_ix2 = i + n_steps
        end_l2 = i+n_steps+lag
        if end_l2 >= len_datos:
            break
        seq_x2 = datos[i:end_ix2, : ]
        seq_y2 = datos[end_l2][0]
        X_test.append(seq_x2)
        y_test.append(seq_y2)

    return np.array(X), np.array(y),np.array(X_test), np.array(y_test)



# %% Definición de los parámetros del modelo.

batch_size = 1
n_steps = 6 # Longitud de la ventana temporal.
lag = 1 # Horizonte Temporal a predecir.
neurons = 5 # Cantidad de Neuronas de la capa LSTM
n_epochs = 100 # Cantidad de iteraciones.

# X_train, y_train, X_test, y_test = SecuenciasTrainingTest(data_scaled,n_steps,pc_train = 0.8,lag = lag)
X_train, y_train, X_test, y_test = SecuenciasTrainingTest(df_all,n_steps,pc_train = 0.8,lag = lag)

ini = len(X_train) # Comienzo del set de evaluación.
ini_val = df.iloc[ini,0] # Valor inicial de precios

# %% Entrenamiento

def fit_lstm_stateful(X_train,y_train,batch_size,n_steps, n_epochs, neurons, lag):
    """ Definición del modelo """
    History_list = []
    model = Sequential() # Capa de entrada
    model.add(LSTM(neurons, batch_input_shape = (batch_size, X_train.shape[1],X_train.shape[2]), stateful = True)) # Capa LSTM
    # model.add(Dropout(0.1)) # Capa Dropout para introducir variaciones aleatorias.
    model.add(Dense(1)) # Capa de salida
    model.compile(loss = 'mse', optimizer = 'adam',metrics=['mse', 'acc']) # Compilador


    """ Comienzo del entrenamiento """
    print("BEGIN TRAINING")
    start_time = time.time()
    for i in range(n_epochs):
        if i >= 1:
            if i == 1:
                elapsed_time = time.time() - start_time
            print("Epoch = ",i+1,"/",n_epochs," | ETA = ",round( elapsed_time * (n_epochs - i)  / 60,2) ," (min)")
        History = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        History_list.append(History)
        model.reset_states()
    return model, History_list



model, History = fit_lstm_stateful(X_train,y_train, batch_size,n_steps, n_epochs, neurons, lag )

# %% Predicciones
# En esta sección compararemos a las predicciones del modelo con los datos reales.




def Forecast_lstm(X_test,model,scaler):
    """ Crea predicciones basadas en el modelo sin reentrenamiento """
    Predictions = list()
    print("Forecasting LSTM Model")
    for i in range(len(y_test)):
        x_input = X_test[i].reshape(1,X_test.shape[1],X_test.shape[2])
        yhat = model.predict(x_input,verbose = 0)
        # yhat = scaler.inverse_transform(yhat.reshape(1,-1))
        Predictions.append(yhat[0][0])
    # Predictions = [Predictions[i][0][0] for i in range(len(Predictions))]
    # Predictions = np.asarray(Predictions)
    return Predictions



def InverseScale(vec,scaler):
    """ Reescala las predicciones a la escala original """
    vec_rescaled = list()
    for i in range(len(vec)):
        inverted = scaler.inverse_transform(vec[i].reshape(1,-1))
        vec_rescaled.append(inverted[0][0])
    return np.array(vec_rescaled)


Predictions = Forecast_lstm(X_test,model,scaler) # Predicciones

# Real_Predictions = InverseScale(Predictions,scaler) # Predicciones reescaladas
# Real_Output = InverseScale(y_test,scaler) # Salida real.
Real_Output = y_test
# %% Predicciones reentrenadas
# Esta sección crea predicciones pero irá reentrenando el modelo a medida que las reciba.

def UpdateLSTM(model,X,y,n_epochs_re):
    """ Reentrena al modelo con las nuevas muestras """
    model.fit(X,y,epochs = n_epochs_re, verbose = 0,batch_size=batch_size)
    return model




len_test = len(y_test)
Predictions2 = [] # Nuevas predicciones
start_time = time.time()
for i in range(len_test):
    if i == 1:
        elapsed_time = time.time() - start_time
    if i>= 1:
        print("Muestra N°",i,"/",len_test,". | ETA = ",round( elapsed_time * (len_test - i)  / 60,2) ," (min)")
    x_input = X_test[i].reshape(1,X_test.shape[1],X_test.shape[2])
    yhat = model.predict(x_input,verbose = 0) # Predicción.
    # yhat = scaler.inverse_transform(yhat.reshape(1,-1)) # Reescalamiento.
    Predictions2.append(yhat[0][0])
    out = np.array(y_test[i].reshape(1,1))
    model = UpdateLSTM(model,x_input, out, 1) # Updateo del modelo.


# %% RMSE


RMSE_sin = mean_squared_error(Real_Output,Predictions)
RMSE_con = mean_squared_error(Real_Output,Predictions2)
print("RMSE SIN :",round(RMSE_sin,6)," | RMSE CON: ",round(RMSE_con,6))


# %% Escalamiento de Predictions solo para el porcentaje
Predictions_scaled = list()
Real_scaled = list()
for i in range(len(Predictions2)):
    Real_scaled.append(round(Real_Output[i]*100,2))
    Predictions_scaled.append(round(Predictions2[i]*100,2))
    # Predictions_scaled.append(round(Real_Output[i]*100+1*np.random.randn(),2))


# %% Gráficos de comparación.
fechas = df.index.values[ini:(df.shape[0]-1)]

fsize = 20

plt.title('Retornos semanales de SPY vs Predicción : 100 epochs',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
# plt.plot(Real_Output,label = 'Valor Real')
plt.plot(fechas[:len(Real_scaled)],Real_scaled,label = 'Valor Real',linewidth=2)
plt.xlabel('Fecha',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('Retorno (%)',fontsize=fsize)
plt.yticks(fontsize=fsize)
# plt.plot(y_test,label = 'Valor Real')
# plt.plot(Predictions,label = 'Predicción')
# plt.plot(Predictions2,label = 'Predicción Reentrenada')
# plt.plot(Predictions2,label = 'Predicción') # Es la reentrenada esta igual
plt.plot(fechas[:len(Real_scaled)],Predictions_scaled,label = 'Predicción',linewidth=2,path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()]) # Es la reentrenada esta igual
# plt.plot(rec2,label = 'Predicción Filtrada')
plt.legend(fontsize = fsize)
plt.show()

# %% Reconstrucción de Precios
# Necesito saber cual es el valor inicial
ini = len(X_train)
ini_og = ini
precios_sin = list()
precios_con = list()

# precios_sin.append(ini_val)
ini = ini_og + n_steps
for i in range(len(y_test)-1):
    precios_sin.append(df.iloc[ini+i,0]*(1+Predictions[i]))
    precios_con.append(df.iloc[ini+i,0]*(1+Predictions2[i]))
    # precios_sin.append(precios_sin[i]*(1+Predictions2[i]))


true_vals_precio = df.iloc[ini:(df.shape[0]-1),0].values
fechas = df.index.values[ini:(df.shape[0]-1)]



# %% Figura de reconstruicción de precios

fsize = 20
plt.figure()
plt.title('Precio de SPY vs Predicción : 100 epochs',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(fechas[:len(precios_con)],true_vals_precio[:len(precios_con)],label = 'Valor Real',linewidth=3,path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
plt.xlabel('Fecha',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('SPY (USD)',fontsize=fsize)
plt.yticks(fontsize=fsize)

# plt.plot(precios_sin,label = 'Predicción')
plt.plot(fechas[:len(precios_con)],precios_con,label = 'Predicción',linewidth=2,path_effects=[pe.Stroke(linewidth=1, foreground='r'), pe.Normal()]) # Reentrenada
# plt.plot(rec2,label = 'Predicción Filtrada')
plt.legend(fontsize = fsize)
plt.show()

# %%

RMSE_precios = mean_squared_error(true_vals_precio[:len(precios_con)],precios_con)

