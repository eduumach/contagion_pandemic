import pandas as pd
import numpy as np


def contagion(days):
    try:
        data = pd.read_csv('dados.csv')
    except:
        print("please download the file")
        quit()

    data['date'] = pd.to_datetime(data['date'])

    date = data['date'].values
    positive = data['positive_rate'].values

    dados = pd.DataFrame({'date': date, 'positive_rate': positive})

    dados = dados.groupby(['date'])['positive_rate'].sum().reset_index()

    x = np.array(dados['date'].index.tolist())
    y = dados['positive_rate'].values

    mymodel = np.poly1d(np.polyfit(x, y, 3))

    days_array = []
    for day in range(days):
        days_array.append(mymodel(day))

    return days_array


def print_contagion(days):
    if days < 0:
        print("please put a number greater than 0")
    else:
        i = 1
        for case in contagion(days):
            print("day: ", i, "number of cases:", case)
            i += 1
