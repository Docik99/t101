import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

#Листинг 1 – Генерация данных линейной регрессии
def lin_gen(a, b, n):
    x = 2 * np.random.rand(n, 1)
    y = b + a * x + np.random.randn(n, 1)
    return x, y

#Листинг 2 – Генерация данных полиномиальной регрессии
def pol_gen(a, b, c, d, n):
    x = 6 * np.random.rand(n, 1) - 3
    t = list(x)
    t.sort()
    x = np.array(t)
    y = a * x ** 3 + b * x ** 2 + c * x + d + np.random.randn(n, 1)
    return x, y

# 2.	Нормальное решение линейной регрессии
# Листинг 3 – Нормальное решение
def lin_reg(cor_x, cor_y):
    n = len(cor_y)
    x = np.c_[np.ones((n, 1)), cor_x]
    teta = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(cor_y)
    plt.plot(x.transpose()[1], cor_y, "b.")
    new_cor_x = np.array([[0], [100]])
    new_x = np.c_[np.ones((2, 1)), new_cor_x]
    new_y = new_x.dot(teta)
    new_x = [[0], [100]]
    plt.plot(new_x, new_y, "r-")
    plt.axis([0, 2, 0, 10])
    plt.show()
    print(teta)

#Листинг 4 – Градиентный спуск
def gradient_down(cor_x, cor_y):
    plt.plot(cor_x, cor_y, "b.")
    plt.axis([0, 2, 0, 10])
    eta = 0.1
    m = len(cor_y)
    n = 1000
    teta = np.random.randn(2, 1)
    x = np.c_[np.ones((m, 1)), cor_x]
    for iteration in range(n):
        gradients = (2 / m) * x.transpose().dot(x.dot(teta) - cor_y)
        teta = teta - eta * gradients
        y2 = x.dot(teta)
    plt.plot(x, y2, "r")
    plt.show()
    print(teta)

#Листинг 5 – Мини-пакетный градиентный спуск
def mini_gradient_down(x, y, n):
    size_batch = n // 100
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    size = len(y)
    x_b = np.c_[np.ones((int(size - 0.2 * size), 1)), x_train]
    n_epochs = 50
    m = int(size - 0.2 * size)
    teta = np.random.rand(2, 1)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(size - size_batch)
            xi = x_b[random_index:random_index + size_batch]
            yi = y_train[random_index:random_index + size_batch]
            gradients = 2/size_batch * xi.T.dot(xi.dot(teta) - yi)
            eta = learning_shedule(epoch * m + i)
            teta = teta - eta * gradients
            t = list(x_test)
    t.sort()
    x_test = np.array(t)
    x_new_b = np.c_[np.ones((int(0.2 * size), 1)), x_test]
    y_predict = x_new_b.dot(teta)
    plt.plot(x_test, y_predict, "r-")
    plt.plot(x_train, y_train, "b.")
    plt.show()
    print(teta)


#Листинг 6 – Стохастический градиентный спуск
def sgd(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    size = len(y)
    x_b = np.c_[np.ones((int(size - 0.2 * size), 1)), x_train]
    n_epochs = 50
    m = int(size - 0.2 * size)
    teta = np.random.rand(2, 1)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = x_b[random_index:random_index + 1]
            yi = y_train[random_index:random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(teta) - yi)
            eta = learning_shedule(epoch * m + i)
            teta = teta - eta * gradients
    t = list(x_test)
    t.sort()
    x_test = np.array(t)
    x_new_b = np.c_[np.ones((int(0.2 * size), 1)), x_test]
    y_predict = x_new_b.dot(teta)
    plt.plot(x_test, y_predict, "r-")
    plt.plot(x_train, y_train, "b.")
    plt.show()
    print(teta)


def learning_shedule(t):
    t0, t1 = 5, 50
    return t0 / (t + t1)


#Листинг 7 –Полиномиальная регрессия
def polin_regr(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    size = len(y)
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly_features.fit_transform(x_train)
    x_b = np.c_[np.ones((int(size - 0.2 * size), 1)), x_poly]
    teta = batch_gradient_down_pr(x_b, y_train)
    t = list(x_test)
    t.sort()
    x_test = np.array(t)
    x_poly = poly_features.fit_transform(x_test)
    x_new_b = np.c_[np.ones((int(0.2 * size), 1)), x_poly]
    y_predict = x_new_b.dot(teta)
    plt.plot(x_test, y_predict, "r-", linewidth=3)
    plt.plot(x, y, "b.", linewidth=2)
    plt.show()
    polynomial_regression = Pipeline(
        [("poly_features", PolynomialFeatures(degree=3, include_bias=False)), ("lin_reg", LinearRegression()), ])
    plot_learning_curves(polynomial_regression, x, y)
    print(teta)

# 5.	Анализ полиномиальной регрессии
# Листинг 8 – Функция для построения кривых обучения
def plot_learning_curves(model, x, y):
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="val")
    plt.show()


#Листинг 10 – Функция для построения J для обучающей и валидационной выборок при разной степени аппроксимирующего полинома (гепперпараметра d)
def gipper_d(x, y):
    parametrs_d = np.arange(2, 5, 1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    tr_errors = []
    val_errors = []
    for degree in parametrs_d:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features.fit(x)
        x_poly_tr = poly_features.fit_transform(x_train)
        x_poly_val = poly_features.fit_transform(x_val)
        model = Pipeline(
            [("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
             ("lin_reg", LinearRegression()), ])
        model.fit(x_poly_tr, y_train)
        predicted_y_train = model.predict(x_poly_tr)
        predicted_y_val = model.predict(x_poly_val)
        tr_errors.append(mean_squared_error(y_train, predicted_y_train))
        val_errors.append(mean_squared_error(y_val, predicted_y_val))
    plt.plot(parametrs_d, np.sqrt(tr_errors), 'r')
    plt.plot(parametrs_d, np.sqrt(val_errors), 'b')
    plt.show()


if __name__ == '__main__':
    lin_x, lin_y = lin_gen(1, 3, 100)
    pol_x, pol_y = pol_gen(2, 0.2, -0.3, 2, 100)
    lin_reg(lin_x, lin_y)
    gradient_down(lin_x, lin_y)
    mini_gradient_down(lin_x, lin_y, 100)
    sgd(lin_x, lin_y)
    plot_learning_curves(,pol_x, pol_y)