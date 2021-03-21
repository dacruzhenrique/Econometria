import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import statistics as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#Regressão Múltipla - X2 com erro de medida
# X2 e X3 independentes
#Vamos supor que Var(X) = Var(w) = 1

B1 = []
B2 = []
B3 = []

for i in range(0,1000):

    X2 = (np.random.normal(loc=0,scale=1,size=100)+10)
    X3 = (np.random.normal(loc=0,scale=1,size=100)+20)
    w = np.random.normal(loc=0,scale=1,size=100)
    u = np.random.normal(loc=0,scale=1,size=100)
    X2e = X2 + w
    
    Y = 5 + 5*X2 + 5*X3 + u
    
    #cria um df que recebe as duas litas, x2e e x3, essas são as variaveis independentes
    df = pd.DataFrame(list(zip(X2e,X3)))
    df.columns = ["X2e","X3"]

    #cria uma var Xcodigo que
    Xcodigo = df[['X2e','X3']]
    Xcodigo = sm.add_constant(Xcodigo)

    #criando o modelo de reg.múltipla
    mod = sm.OLS(endog=Y, exog=Xcodigo)
    res = mod.fit()
    B1.append(res.params[0])
    B2.append(res.params[1])
    B3.append(res.params[2])

sns.distplot(B1)
plt.show()