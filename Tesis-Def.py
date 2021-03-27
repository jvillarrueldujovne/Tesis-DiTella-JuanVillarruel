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


tickers = ['SPY','GLD','QQQ','VWO','XLV','VNQ','XLE','XLP','XLF','XLU',"^TNX","^VIX"] # Lista de Tickers.

df = [] # En este dataframe se almacenarán todas las series temporales de los datos.
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

# %% La tasa de 10 años (^TNX) tenía ciertos valores faltantes 'nan'.
# Esta sección los rellena con el valor anterior.
# Además, conviertes los datos finales a un 'ndarray' para un procesamiento más ágil.

a = df['^TNX'].isnull()
df3 = cp.deepcopy(df)

vals = list()
i=0
for values in df.index.values:
    if np.isnan(df.loc[values]['^TNX']):
        print(df.loc[values]['^TNX'])
        df.loc[values,'^TNX'] =  df.loc[df3.index.values[i-1],'^TNX']
    i+=1


df_all = df.rename_axis('ID').values # Convierto a un ndarray
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




# %% Definición de los parámetros del modelo y separación de datos.
# Esta función creará los arrays de entrada y salida del modelo, además de separar
# los datos en las series de entrenamiento y evaluación.

# Las dimensiones de la entrada serán (1,q_activos,n_steps).
# El vector de entrenamiento será un vector de (len_datos*pc_train,q_activos,n_steps).




def SecuenciasTrainingTest(datos,n_steps,pc_train = 0.9,lag=1):
    # datos es un DataFrame con los datos.
    # n_steps es la cantidad de días usados para predecir.
    # dias_delay es la cantidad de días para adelante que quiero predecir
    X , y = list(), list()
    len_datos = datos.shape[0]-lag-1
    # last_col = datos.shape[1]-1
    # inpt = datos[0:len_datos , :-1] # El input son los datos menos la última columna
    for i in range(round(len_datos*pc_train)):
        print(i)
        # Me fijo donde está el final de la secuencia:
        # seq_x = list()
        seq_x, seq_y = list(), list()
        end_ix = i + n_steps
        end_l = i+n_steps+lag
        if end_l >= len_datos:
            break
        seq_x = datos[i:end_ix, : ]
        seq_y = datos[end_l][0]
        X.append(seq_x)
        y.append(seq_y)


    X_test, y_test = list(),list()
    print("SWITCH")
    for i in range(round(len_datos*pc_train),len_datos):

        print(i)
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



# Definición de los parámetros del modelo.

batch_size = 1
n_steps = 50 # Longitud de la ventana temporal.
lag = 1 # Horizonte Temporal a predecir.
neurons = 100 # Cantidad de Neuronas de la capa LSTM
n_epochs = 50 # Cantidad de iteraciones.

X_train, y_train, X_test, y_test = SecuenciasTrainingTest(data_scaled,n_steps,pc_train = 0.8,lag = lag)

# %% Definición del modelo y entrenamiento.


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
        yhat = scaler.inverse_transform(yhat.reshape(1,-1))
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

Real_Predictions = InverseScale(Predictions,scaler) # Predicciones reescaladas
Real_Output = InverseScale(y_test,scaler) # Salida real.








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
    yhat = scaler.inverse_transform(yhat.reshape(1,-1)) # Reescalamiento.
    Predictions2.append(yhat[0][0])
    out = np.array(y_test[i].reshape(1,1))
    model = UpdateLSTM(model,x_input, out, 1) # Updateo del modelo.

# %% Correción temporal de las predicciones para que coincidan con la muestra determinada.

for i in range(lag):
    Predictions.pop(0)
    Predictions2.pop(0)


# %% Gráficos de comparación.
fsize = 20

plt.title('SPY vs Predicción : 100 epochs',fontsize=fsize)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(Real_Output,label = 'Valor Real')
plt.xlabel('Muestra',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('SPY (USD)',fontsize=fsize)
plt.yticks(fontsize=fsize)
# plt.plot(y_test,label = 'Valor Real')
plt.plot(Predictions,label = 'Predicción')
plt.plot(Predictions2,label = 'Predicción Reentrenada')
# plt.plot(rec2,label = 'Predicción Filtrada')
plt.legend(fontsize = fsize)
plt.show()

# %% Calculo de Error Cuadrático Medio con y sin reentrenamiento.

RMSE_sin = mean_squared_error(Real_Output[0:len(Predictions)],Predictions)
RMSE_con = mean_squared_error(Real_Output[0:len(Predictions2)],Predictions2)


# %% Cálculo de histograma de los retornos típicos con el horizonte determinado.

# Cálculo de retornos.
Retornos_reales = list()
horizonte = lag
values = df['SPY'].tolist()

for i in range(len(values)-horizonte):
    ret = round(((1 - values[i+horizonte] / values[i] )*100),2)
    Retornos_reales.append(ret)
    
    
ret_mean = np.mean(Retornos_reales) # Media de los retornos.
ret_std = np.var(Retornos_reales)**0.5 # Desvío estándar.

# %% Histograma de retornos.
q_bins = 1000
factor = 100
ret_mean = np.mean(Retornos_reales)
ret_std = np.var(Retornos_reales)**0.5



fsize = 20
plt.xlabel('Retorno (%)',fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('Cantidad de apariciones',fontsize=fsize)
plt.yticks(fontsize=fsize)
# Create the bins and histogram
count, bins, ignored = plt.hist(Retornos_reales, q_bins, density = False)
plt.title('Histograma de retornos "SPY" luego de 1 día',fontsize = 28)
# Plot the distribution curve
plt.plot(bins, factor * 1/(ret_std * np.sqrt(2 * np.pi)) *
    np.exp( - (bins - ret_mean)**2 / (2 * ret_std**2) ),       linewidth=3, color='y')
plt.show()



# %% Estrategia : Cálculo de Retornos teóricos.


Retornos = list() # Cálculo de retornos teóricos según nuestras predicciones
horizonte = lag

values = Predictions2 # Predicciones Reentrenadas
values_real = df['SPY'].tolist()
values_real = values_real[-len(values):] # Valores reales.
 




# Cálculo de retornos
for i in range(len(values)-horizonte):
    ret = round(((1 - values[i+horizonte] / values_real[i] )*100),2)
    Retornos.append(ret)


# %% Estrategia de inversión.


benchmark = 1.08 # Retorno esperado a alcanzar.
Capital_Inicial = 100000
Cantidad_acciones = 0
Capital = Capital_Inicial
Block = False
# lag = 5

last_i = 0
last_cap = 0
q_trades = 0
wins = 0
win_list = []
losses = 0
loss_list = []
neutral = 0


for i in range(len(Retornos)):
    if Retornos[i] > benchmark:
        if Block == False:
            Block = True
            precio = round(values_real[i+lag],2)
            Cantidad_acciones += np.floor(Capital/precio)
            last_cap = Capital
            Capital -= Cantidad_acciones * precio
            last_i = i
            q_trades += 1
            print("Bougth",Cantidad_acciones," @ ", precio," | i =",i)
            
# < >    
    if Block == True:
        if i - last_i >= lag:
            precio = round(values_real[i+lag],2)
            Capital += precio * Cantidad_acciones
            print("Sold",Cantidad_acciones," @ ", precio," | i =",i)
            Cantidad_acciones -= Cantidad_acciones
            Block = False

            if last_cap > Capital:
                losses += 1
                print("LOSS of ",last_cap - Capital)
                loss_list.append(last_cap - Capital)
            elif last_cap < Capital:
                wins += 1
                print("WIN of ", Capital - last_cap)
                win_list.append(Capital - last_cap)
            else:
                neutral += 1
                print("Neutral Trade")

            print("Cap =", Capital)
            print("-----------------------------------------------")      
if Cantidad_acciones != 0:
    Capital += values_real[i+lag] * Cantidad_acciones
    print("Sold",Cantidad_acciones," @ ", values_real[i+lag])
    Cantidad_acciones -= Cantidad_acciones
    Block = False
    print("Cap =", Capital)
    print("-----------------------------------------------")
    

PnL = Capital - Capital_Inicial
Retorno = PnL / Capital_Inicial * 100
print("Inicio : ", Capital_Inicial," | Final : ", round(Capital,2))
print("P&L : ",round(PnL,2))
print("Retorno : ",round(Retorno,2)," %")
print("Cantidad de Trades : ",q_trades)
print("Wins : ",wins," | Losses : ",losses," | Neutral : ",neutral)
print("Win % :",round(wins/(wins+losses+neutral) * 100,2))
print("Win Average :",round(np.mean(win_list),2)," | Loss Average :", round(np.mean(loss_list),2))