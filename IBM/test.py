import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import csv

data = 'C:\\Users\Bernardo Duarte\Documents\Machine-Learning\IBM\FuelConsumptionCo2.csv'

eng_size = []
co2_emiss = []

with open(data, mode='r') as file:  # abre o arquivo no modo leitura
    # cria uma variável para receber a leitura do arquivo
    file_csv = csv.reader(file)
    next(file_csv)  # pula o cabeçalho

    for line in file_csv:  # itera sobre cada linha da variavel
        eng_size.append(float(line[4]))
        co2_emiss.append(float(line[12]))
        # extrai os dados transformanddo em float

# Gráfico de dispersão ENGINESIZE vs CO2EMISSIONS
plt.figure(1)
plt.scatter(eng_size, co2_emiss, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

eng_size_np = np.array(eng_size).reshape(-1, 1)  # matriz 2D
co2_emiss_np = np.array(co2_emiss)  # array 1D

X_train, X_test, y_train, y_test = train_test_split(
    eng_size_np, co2_emiss_np, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

plt.figure(2)
plt.scatter(eng_size, co2_emiss, color='blue')
plt.scatter(X_train, y_train, color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

a = model.coef_[0]
b = model.intercept_

# Imprime os coeficientes
print('Coeficiente Angular: ', a)
print('Coeficiente Linear: ', b)

# Plota a linha de regressão
plt.figure(3)
plt.scatter(eng_size, co2_emiss, color='blue')
plt.plot(X_train, a * np.array(X_train) + b, color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Linear Regression: Engine size vs Emission")
plt.show()

# Avaliação do modelo
y_predict = model.predict(X_test)

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predict - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_predict - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test, y_predict))
