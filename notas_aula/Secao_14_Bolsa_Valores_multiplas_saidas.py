#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 19:50:07 2018

@author: jrperin
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('petr4-treinamento.csv')
base = base.dropna()

#Abertura e Maximo (high)
base_treinamento = base.iloc[:, 1:2].values
base_valor_maximo = base.iloc[:, 2:3].values

# Fazer a normalização dos dados
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)


previsores = []
preco_real1 = []
preco_real2 = []

for i in range(90, 1242):
    #vamos usar apenas o Open como atributo previsor
    previsores.append(base_treinamento_normalizada[i -90 : i, 0])
    preco_real1.append(base_treinamento_normalizada[i, 0])
    preco_real2.append(base_valor_maximo_normalizada[i, 0])

#Converte de Lista para NumpyArray
previsores, preco_real1, preco_real2 = np.array(previsores), np.array(preco_real1), np.array(preco_real2)

# Reshape - pq só temos 1 atributo previsor
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

#Vamos criar um unico dataframe
preco_real = np.column_stack((preco_real1, preco_real2))


regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense (units = 2,activation = 'linear'))

# Alterado para adam
regressor.compile(optimizer = 'rmsprop', loss= 'mean_squared_error', metrics = ['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)


# vamos carregar a base de testes
base_teste = pd.read_csv('petr4-teste.csv')
preco_real_open = base_teste.iloc[: , 1:2].values
preco_real_high = base_teste.iloc[: , 2:3].values


base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90: i, 0])
#Converter para NunpyArray
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)


plt.plot(preco_real_open, color = 'red', label = 'Preco Abertura Real')
plt.plot(preco_real_high, color = 'black', label = 'Preco Alta Real')
plt.plot(previsoes[:, 0], color = 'blue', label = 'Previsoes Abertura')
plt.plot(previsoes[:, 1], color = 'orange', label = 'Previsoes Alta')
plt.title('Previsao Preco das Acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()