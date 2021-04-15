### Análise do desepenho dos alunos de ensino médio nos EUA

#Importando as bibliotecas
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Carregando o dataset
df = pd.read_excel('C:/Users/00hcr/Documents/datasets/StudentsPerformance.xlsx')

#Visualizando o dataframe
df.head()

#Alterando o nome de algumas colunas
df = df.rename(columns = {'gender':'gênero'})
df = df.rename(columns = {'lunch':'almoço'})
df = df.rename(columns = {'test preparation course':'curso_prep'})
df.head()

#Forma do df
df.shape

#Descrição das variáveis numéricas
df.describe()

#Resumo dos valores de gênero
df["gênero"].value_counts()

#Resumo dos valores de escolaridade dos pais
df["parental level of education"].value_counts()

#Resumo dos valores de almoço
df['almoço'].value_counts()

#Resumo dos dos valores de curso preparatório
df['curso_prep'].value_counts()

#Verificando a exitência de dados missing
df.isnull().sum()

#Histograma com a distribuição das notas do teste de matemática
plt.hist(df['math score'])

#Histograma com a distribuição das notas do teste de leitura
plt.hist(df['reading score'])

#Histograma com a distribuição das notas do teste de escrita
plt.hist(df["writing score"])

#Quantos estudantes foram aprovados no teste de matemática?
media = 60
df['resultado_mat'] = np.where(df['math score']>media, 'A', 'R')
df.resultado_mat.value_counts()

#Quantos estudantes foram aprovados no teste de leitura?
df['resultado_leitura'] = np.where(df['reading score']>media, 'A', 'R')
df.resultado_leitura.value_counts()

#Quantos estudantes foram aprovados no teste de escrita?
df['resultado_escrita'] = np.where(df['writing score']>media, 'A', 'R')
df.resultado_escrita.value_counts()
df.head()

#Criando uma coluna que é uma representação numérica dos níveis de escolaridade dos pais
lista = []
for i in range(0,len(df['parental level of education'])):
    if df.iloc[i,1] == "some high school":
        lista.append(1)
    
    elif df.iloc[i,1] == "high school":
        lista.append(2)

    elif df.iloc[i,1] == "associate's degree":
        lista.append(3)

    elif df.iloc[i,1] == "some college":
        lista.append(4)
    
    elif df.iloc[i,1] == "bachelor's degree":
        lista.append(5)

    else:
        lista.append(6)

df.insert(loc=2,column='escolaridade_pais',value=lista)

#Analisando a média de nota em cada de teste, de acordo com nível de escolaridade dos pais
data = df[['escolaridade_pais','math score','reading score','writing score']].groupby('escolaridade_pais')['math score', 'reading score', 'writing score'].mean().agg(lambda x: list(set(x))).reset_index()
print(data)

#Criando uma coluna que é a representação numérica dos resultados dos testes
# Aprovado = 1 ; Reprovado = 0

#Matemática
valores1 = []

for i in range(0,len(df['resultado_mat'])):
        
        if df.iloc[i,8] == 'A':
            valores1.append(1)
        else:
            valores1.append(0)
df.insert(loc=11,column='matemática',value=valores1)

#Leitura
valores2 = []

for i in range(0,len(df['resultado_leitura'])):
        
        if df.iloc[i,9] == 'A':
            valores2.append(1)
        else:
            valores2.append(0)
df.insert(loc=12,column='leitura',value=valores2)

#Escrita
valores3 = []

for i in range(0,len(df['resultado_mat'])):
        
        if df.iloc[i,10] == 'A':
            valores3.append(1)
        else:
            valores3.append(0)
df.insert(loc=13,column='escrita',value=valores3)

#Retirando as colunas que foram substituídas pelo respectivo valor numérico
df.drop(['parental level of education', 'resultado_mat', 'resultado_leitura', 'resultado_escrita'], 
        axis=1, inplace = True)

#Visualizando a porcentagem de aprovados em cada teste, agrupados por curso preparatório
data = df[['curso_prep', 'matemática', 'leitura', 'escrita']].groupby('curso_prep')['matemática', 'leitura', 'escrita'].mean().agg(lambda x: list(set(x))).reset_index()
data

### Regressão Logística ###

# i) Modelo para o teste de matemática

# Transforma classe em categorico
df['escolaridade_pais'] = df['escolaridade_pais'].astype('category')

# Modelo para o teste de matemática
modelo1 = smf.glm(formula='matemática ~ gênero + escolaridade_pais + almoço + curso_prep' , data=df,
                family = sm.families.Binomial()).fit()
print(modelo1.summary())

#Facilita a interpretação
print(np.exp(modelo1.params[1:]))


# ii) Modelo para o teste de leitura 

modelo2 = smf.glm(formula='leitura ~ gênero + escolaridade_pais + almoço + curso_prep' , data=df,
                family = sm.families.Binomial()).fit()
print(modelo2.summary())

#Facilita a interpretação
print(np.exp(modelo2.params[1:]))


# iii) Modelo para o teste de escrita 

modelo3 = smf.glm(formula='escrita ~ gênero + escolaridade_pais + almoço + curso_prep' , data=df,
                family = sm.families.Binomial()).fit()
print(modelo3.summary())

#Facilita a interpretação
print(np.exp(modelo3.params[1:]))


# Muito obrigado! # 
