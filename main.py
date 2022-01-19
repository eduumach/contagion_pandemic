import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


dados = pd.read_csv('dados.csv')
#print(dados.dtypes)

dados['date'] = pd.to_datetime(dados['date'])

#dados_ordenados = dados.sort_values(by='date')

data = dados['date'].values
positivo = dados['positive_rate'].values

dados = pd.DataFrame({'date':data, 'positive_rate':positivo})

dados = dados.groupby(['date'])['positive_rate'].sum().reset_index()

X = np.array(dados['date'].index.tolist())
Y = dados['positive_rate'].values


plt.scatter(X, Y)
#plt.show()

r = pearsonr(X, Y)
print(f'Coeficiente de correlação: {r}') 

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)

reg = LinearRegression()
reg.fit(x_train,y_train)
pred = reg.predict(x_test)

r_squared = r2_score(y_test, pred)
print(f'Coeficiente r2: {r_squared}')

residual = y_test - pred

plt.scatter(X, Y, color="blue")
plt.plot(x_test, pred, color="red")
plt.title("Testes positivos covid-19")
plt.xlabel("Numero de dias")
plt.ylabel("numero de casos")
plt.hist(residual, rwidth=0.9)
plt.show()