import numpy as np
import matplotlib.pyplot as plt

from lab02.parser import csv_reader, files_name


def cal_cost(theta, X, y):
    """
    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))

    where:
        j is the no of features
    """

    m = len(y)

    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    """
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)

        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)

    return theta, cost_history, theta_history


if __name__ == '__main__':
    lr = 0.1
    n_iter = 1000

    files = files_name()
    first_data = csv_reader(files)
    X = np.array([first_data['mileage']/10000, ]).T
    y = np.array([first_data['price']/10000, ]).T
    # X = 2 * np.random.rand(100, 1)
    # y = 4 + 3 * X + np.random.randn(100, 1)

    theta = np.random.randn(2, 1)

    X_b = np.c_[np.ones((len(X), 1)), X]
    theta, cost_history, theta_history = gradient_descent(X_b, y, theta, lr, n_iter)

    print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0], theta[1][0]))
    print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    _ = ax.plot(range(n_iter), cost_history, 'b.')
    plt.show()
    exit(0)
