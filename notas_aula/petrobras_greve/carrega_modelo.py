#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 00:09:47 2018

@author: jrperin
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from keras.models import model_from_json


# load json and create model
json_file = open('petr4_regressor.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("petr4_regressor.h5")
print("Loaded model from disk")


interv = 90

# Base
base = pd.read_csv('petr4_treinamento_ex.csv')
# Excluir valores nulos
base = base.dropna()
# Seleciona todas as linhas e coluna 1 (Open)
base_treino = base.iloc[: , 1:2].values

# Normalizar a base entre 0 e 1
normalizador = MinMaxScaler(feature_range = (0,1))
base_treino_norm = normalizador.fit_transform(base_treino)


# Previsão dos precos das Acoes
base_teste = pd.read_csv('petr4_teste_ex.csv')

# Seleciona todas as linhas e coluna 1 (Open)
preco_real = base_teste.iloc[:, 1:2].values

# precisamos dos 90 precos anteriores para aplicar o teste
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)

entradas = base_completa[len(base_completa) - len(base_teste) - interv : ].values
# lower bound = 1152, 90 registros antes da base de teste...

# Fazer o reshape
entradas = entradas.reshape(-1, 1) #Fica no formato do numpy
entradas = normalizador.transform(entradas)

X_teste = []
print("Qtd de previsoes = " + str( len(entradas) - interv))
for i in range(interv, len(entradas)):
    X_teste.append(entradas[i - interv : i, 0])
X_teste = np.array(X_teste)

print(X_teste.shape[0])
print(X_teste.shape[1])

X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = loaded_model.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

mean_previsoes = previsoes.mean()
mean_preco_real = preco_real.mean()

print ('Média Previsoes = ' + str(mean_previsoes))

print ('Media Real     = ' + str(mean_preco_real))

print('Previsao - Preco Real = ' + str(mean_previsoes - mean_preco_real))

#plot_real = pd.concat((base['Open'], pd.DataFrame(preco_real)) , axis=0)
#plot_previsao = pd.concat((base['Open'], pd.DataFrame(previsoes)), axis = 0)

plot_real = normalizador.inverse_transform(entradas)
plot_previsoes = np.concatenate((plot_real[0 : -len(previsoes)], previsoes),)

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(plot_real, color = 'red',  label = 'Preco Real') 
plt.plot(plot_previsoes,  color = 'blue', label = 'Previsoes') 
#plt.plot(preco_real, color = 'red',  label = 'Preco Real') 
#plt.plot(previsoes,  color = 'blue', label = 'Previsoes') 
plt.title('Previsao preco das acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()