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
base_treinamento = base.iloc[:, 1:7].values

# Fazer a normalização dos dados
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizado = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizado[i -90 : i, 0:6])
    preco_real.append(base_treinamento_normalizado[i, 0])

#Converte de Lista para NumpyArray
previsores, preco_real = np.array(previsores), np.array(preco_real)

# Essa linha debaixo nao é mais necessaria, pois os dados já estao no padrao esperado
## previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# Units = 1 -> Significa que vamos prever apenas 1 neuronio, apenas 1 valor...
# Como nossos dados estao normalizados, alteramos para a funcao sigmoid, pois ela retorna valores entre 0 e 1
regressor.add(Dense (units = 1,activation = 'sigmoid'))

# Alterado para adam
regressor.compile(optimizer = 'adam', loss= 'mean_squared_error', metrics = ['mean_absolute_error'])


es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)

#Permitir que o Keras salve os modelos com os melhores resultados
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', save_best_only = True, verbose = 1)

regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32, callbacks = [es, rlr, mcp])

