#Importando as bibliotecas utilizadas
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt
import statistics as st
from random import gauss

#Crinado as variáveis e rodando a regressão com X2 possuindo um erro de medida w
B2 = []
for i in range(0,1000):

    u = np.random.normal(loc=0,scale=1,size = 100)
    w = np.random.normal(loc=0,scale=1,size = 100)
    X2 = (np.random.normal(loc=0,scale=1,size = 100) + 10)
    X3 = (np.random.normal(loc=0,scale=1,size = 100) + 20)
    X2e = X2 + w

    Y = 5 + 5*X2 + 5*X3 + u

#cria um df que recebe as duas litas, x2e e x3, essas são as variaveis independentes
    df = pd.DataFrame(list(zip(X2e,X3)))
    df.columns = ['X2e','X3']

#cria uma var Xcodigo que 
    Xcodigo = df[['X2e','X3']]
    Xcodigo = sm.add_constant(Xcodigo)

    mod = sm.OLS(endog = Y,exog = Xcodigo)
    res = mod.fit()
    B2.append(res.params[1])

#Histograma com os valores estimados para B2
plt.hist(B2)
plt.show()

#Estatísticas descritivas dos valores estimados para B2
print(st.mean(B2), st.stdev(B2), min(B2), max(B2))