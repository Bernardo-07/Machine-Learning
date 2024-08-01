import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import csv

'''data = 'C:\\Users\Bernardo Duarte\Documents\Machine-Learning\IBM\FuelConsumptionCo2.csv'

eng_size = []
co2_emiss = []
fuel_consum = []

with open(data, mode='r') as file: #abre o arquivo no modo leitura
    file_csv = csv.reader(file) #cria uma variável para receber a leitura do arquivo
    next(file_csv) #pula o cabeçalho

    for line in file_csv: #itera sobre cada linha da variavel
        eng_size.append(float(line[4]))
        fuel_consum.append(float(line[10]))
        co2_emiss.append(float(line[12]))
        #extrai os dados transformanddo em float

# Gráfico de dispersão FUELCONSUMPTION_COMB vs CO2EMISSIONS
plt.scatter(fuel_consum, co2_emiss, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# Gráfico de dispersão ENGINESIZE vs CO2EMISSIONS
plt.scatter(eng_size, co2_emiss, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()'''

path="C:\\Users\Bernardo Duarte\Documents\Machine-Learning\IBM\FuelConsumptionCo2.csv"
# Leitura do arquivo CSV
df = pd.read_csv(path)

# Visualizar as primeiras linhas do dataset
print(df.head())

# Resumo dos dados
print(df.describe())

# Seleção de colunas específicas
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))

# Visualização dos dados
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Gráfico de dispersão FUELCONSUMPTION_COMB vs CO2EMISSIONS
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# Gráfico de dispersão ENGINESIZE vs CO2EMISSIONS
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Divisão do conjunto de dados em treinamento e teste
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Visualização dos dados de treinamento
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Criação do modelo de regressão linear
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Coeficientes e intercepto do modelo
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Visualização do modelo ajustado
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Avaliação do modelo
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))