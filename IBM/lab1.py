import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
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