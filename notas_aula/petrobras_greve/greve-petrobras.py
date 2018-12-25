# -*- coding: utf-8 -*-

# Exercício Greve da Petrobras


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from keras.models import model_from_json


base = pd.read_csv('petr4_treinamento_ex.csv')

# Excluir valores nulos
base = base.dropna()

# vamos pegar o preco abertura para treinar (values para ser numpy array)
base_treinamento = base.iloc[: , 1:2].values

# Aplicar uma normalização para ficar na escala de 0 até 1, 
# senao fica muito lenda a 'rede neural recorrente'
normalizador = MinMaxScaler(feature_range =(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)


################### AULA 68 ################################

# Não conseguimos passar a base de dados nesse formato, precisamos prepará-la

# Para trabalhar com uma serie temporal 
# precisamos definir o  << intervalo de tempo >> ex. 4 dias

#    Previsores                   Preco Real
#   19,99 19,80 20,33 20,48         20,11  
#   19,80 20,33 20,48 20,11         19,63
#   20,33 20,48 20,11 19,63         19,77
#   20,48 20,11 19,63 19,77         19,85

# Obs.: Esse é um Problema de aprendizagem supervisionada

# Ideia basica de como vai funciionar.




################### AULA 69 ################################

# Vamos usar 90 dias de intervalo de tempo
interv = 90

previsores = []
preco_real = []

# vamos percorrer a base e adicionar os dados
for i in range(interv, len(base_treinamento_normalizada)):
    previsores.append(base_treinamento_normalizada[i - interv : i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])
    
# precisamos fazer uma transformacao para ficar no formato numpy
    
previsores, preco_real = np.array(previsores), np.array(preco_real)

# Vamos precisar fazer só mais uma alteracao... No Keras Recurrent Layers, o input shape tem tem dimensoes
# -> batch_size, timesteps, input_dim

previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

# Se abrir o previsores no explorador de variaveis, vemos que agora tem 3 eixos

regressor = Sequential()
regressor.add( LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
# Units = células de memoria
# return_sequences = Qdo tem mais de uma camada LSTM
# input_shape = Dados de entrada, 1 = apenas 1 atributo previsor

regressor.add(Dropout(0.3))
# Dropout 30% - zera 30% das entradas para prever overfitting

## Para essa rede neural, o ideal é adicionar mais camadas, senao ela nao funciona direito
regressor.add( LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add( LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add( LSTM(units = 50))
regressor.add(Dropout(0.3))

# ultima camada oculta que estara ligada com a resposta final
regressor.add(Dense(units = 1, activation = 'linear'))
# fazer um teste com a funcao sigmoide depois...

regressor.compile(optimizer='rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])

# rodar no mínimo 100 epocas
regressor.fit(previsores, preco_real, epochs= 100, batch_size = 32)


# Previsão dos precos das Acoes

# Vamos efetivamente fazer as previsoes

base_teste = pd.read_csv('petr4_teste_ex.csv')

# Vamos prever apenas os valores do Open (1.a coluna)
preco_real_teste = base_teste.iloc[:, 1:2].values

# precisamos dos 90 precos anteriores para aplicar o teste
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)


entradas = base_completa[len(base_completa) - len(base_teste) - interv : ].values
# lower bound = 1152, 90 registros antes da base de teste...

# Fazer o reshape
entradas = entradas.reshape(-1, 1) #Fica no formato do numpy
entradas = normalizador.transform(entradas)


X_teste = []
for i in range(interv, len(entradas)):
    X_teste.append(entradas[i - interv : i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)

previsoes = normalizador.inverse_transform(previsoes)

sum_previsoes = previsoes.mean()

sum_preco_real = preco_real_teste.mean()

print('Previsao - Preco Real = ' + str(sum_previsoes - sum_preco_real))



################### AULA 72 ################################
# Grafico com os precos das Acoes

# import matplotlib.pyplot as plt

plt.plot(preco_real_teste, color = 'red',  label = 'Preco Real') 
plt.plot(previsoes,        color = 'blue', label = 'Previsoes') 
plt.title('Previsao preco das acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()


# Salvando o modelo para uso posterior

# sudo pip install h5py
# from keras.models import model_from_json
# import os

# serialize model to JSON
model_json = regressor.to_json()
with open("petr4_regressor.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("petr4_regressor.h5")
json_file.close()
print("Saved model to disk")



# Teste de carga do modelo em: carrega_modelo.py