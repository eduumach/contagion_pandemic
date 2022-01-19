import pandas as pd
import numpy as np


def contagion(day):

    data = pd.read_csv('dados.csv')

    data['date'] = pd.to_datetime(data['date'])

    date = data['date'].values
    positive = data['positive_rate'].values

    dados = pd.DataFrame({'date': date, 'positive_rate': positive})

    dados = dados.groupby(['date'])['positive_rate'].sum().reset_index()

    x = np.array(dados['date'].index.tolist())
    y = dados['positive_rate'].values

    mymodel = np.poly1d(np.polyfit(x, y, 3))

    return mymodel(day)


print(contagion(1))


def print_contagion(days):
    if days < 0:
        print("please put a number greater than 0")
    else:
        for day in range(days):
            print("day: ", day, "number of cases: ", round(contagion(day), 2))
