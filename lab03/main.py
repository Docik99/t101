from random import randint

import matplotlib.pyplot as plt
import numpy as np

from lab03.parser import csv_reader, files_name


def lin_reg(first_data):
    data = {}
    data['x_i'] = first_data['mileage']
    #rand_i = np.random
    data['y_i'] = first_data['price']
    # добавим колонку единиц к единственному столбцу признаков
    X = np.array([np.ones(data['x_i'].shape[0]), data['x_i']]).T
    # перепишем, полученную выше формулу, используя numpy
    # шаг обучения - в этом шаге мы ищем лучшую гипотезу h
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), data['y_i'])
    # шаг применения: посчитаем прогноз
    y_hat = np.dot(w, X.T)
    return data, y_hat


def grafic(data, y_hat):
    print('Shape of X is', data['x_i'].shape)
    print('Head of X is', data['x_i'][:10])

    margin = 0.3
    plt.scatter(data['x_i'], data['y_i'], 40, 'g', 'o', alpha=0.8, label='data')
    plt.plot(data['x_i'], y_hat, 'r', alpha=0.8, label='fitted')
    plt.xlim(data['x_i'].min() - margin, data['x_i'].max() + margin)
    plt.ylim(data['y_i'].min() - margin, data['y_i'].max() + margin)
    plt.legend(loc='upper right', prop={'size': 20})
    plt.title('True manifold and noised data')
    plt.xlabel('MILIAGE')
    plt.ylabel('PRICE')
    plt.show()


if __name__ == '__main__':
    files = files_name()
    first_data = csv_reader(files)
    data, y_hat = lin_reg(first_data)
    grafic(data, y_hat)
