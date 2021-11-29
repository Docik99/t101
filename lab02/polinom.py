import numpy as np
import matplotlib.pyplot as plt


def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    print(np.shape(x))
    print(np.shape(y))
    if len(a) != (n+1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) '
              f'must be the same as polynomial degree {n}')
        return
    for i in range(0, n+1):
        y = y + a[i] * np.power(x, i) + noise*(np.random.rand(size, 1) -0.5)
    print(np.shape(x))
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=', ')


def hypothesis(theta, X, n):
    h = np.ones((X.shape[0], 1))
    theta = theta.reshape(1, n+1)
    for i in range(0, X.shape[0]):
        x_array = np.ones(n+1)
        for j in range(0, n+1):
            x_array[j] = pow(X[i], j)
        x_array = x_array.reshape(n+1, 1)
        h[i] = float(np.matmul(theta, x_array))
    h = h.reshape(X.shape[0])
    return h


def BGD(theta, alpha, num_iters, h, X, y, n):
    theta_history = np.ones((num_iters, n+1))
    cost = np.ones(num_iters)
    for i in range(0, num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1, n+1):
            theta[j] = theta[j]-(alpha/X.shape[0])*sum((h-y)*pow(X, j))
        theta_history[i] = theta
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
    theta = theta.reshape(1, n+1)
    return theta, theta_history, cost


def poly_regression(X, y, alpha, n, num_iters):
    # initializing the parameter vector…
    theta = np.zeros(n+1)
    # hypothesis calculation….
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent
    theta, theta_history, cost = BGD(theta, alpha, num_iters, h, X, y, n)
    return theta, theta_history, cost


if __name__ == '__main__':
    generate_poly([3, 4, 6, 1], 3, 0.001, 'data1.txt')
    data = np.loadtxt('data1.txt', delimiter=', ')
    X_train = data[:, 0]  # the feature_set
    y_train = data[:, 1]  # the labels
    # calling the principal function with learning_rate = 0.0001 and
    # n = 2(quadratic_regression) and num_iters = 300000
    theta, theta_history, cost = poly_regression(X_train, y_train, 0.1, 2, 3000)

    training_predictions = hypothesis(theta, X_train, 2)
    scatter = plt.scatter(X_train, y_train, label="training data")
    regression_line = plt.plot(X_train, training_predictions, label="polynomial(degree 2) regression")
    plt.show()