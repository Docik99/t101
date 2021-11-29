import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


# generates x and y numpy arrays for
# y = a*x + b + a * noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_linear(a, b, noise, filename, size=100):
    # print('Generating random data y = a*x + b')
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b + noise * a * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')
    return (x, y)


def train_test_split_To_file(filename, filename_train, filename_test):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_data = np.hstack((x_train, y_train))
    np.savetxt(filename_train, train_data, delimiter=',')

    test_data = np.hstack((x_test, y_test))
    np.savetxt(filename_test, test_data, delimiter=',')

    return


def create_batch(filename, size_of_batch=0):
    global ran_index
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)
    if size_of_batch == 0:
        size_of_batch = np.shape(x)[0]
    if size_of_batch > 1:
        random_index = np.random.choice(np.shape(x)[0], size_of_batch)
    elif size_of_batch == 1:
        random_index = np.random.choice(np.shape(x)[0])
    else:
        print("Incorrect format size of batch")
    x = x[random_index]
    ones_matrix = np.ones((np.shape(x)[0], 1))
    if size_of_batch == 1:
        X = np.array([[1.], [x[0]]])
    else:
        X = np.concatenate([ones_matrix, x], 1).T
    Y = y[random_index]
    return x, Y, X, Y


# thats an example of linear regression using polyfit
def linear_regression_numpy(filename_train, filename_test, view_img=0):
    # now let's read it back
    with open(filename_test, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)
    # printing shapes is useful for debugging
    # print(np.shape(x))
    # print(np.shape(y))
    # our model
    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)

    time_end = time()
    print(f"[Time] {time_end - time_start} секунд")
    # our hypothesis for give x
    h = model[0] * x + model[1]
    if view_img:
        with open(filename_train, 'r') as f:
            data = np.loadtxt(f, delimiter=',')
        # split to initial arrays
        x_train, y_train = np.hsplit(data, 2)
        # print(exact_model_result)
        h = model[0] * x + model[1]
        # and check if it's ok
        plt.title("Linear regression [polyfit]")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(x_train, y_train, "b.", label='experiment')
        plt.plot(x, h, "r", label='model')
        plt.legend()
        plt.show()
    return model


def linear_regression_exact(filename_train, filename_test, view_img=0):
    with open(filename_test, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)
    ones_matrix = np.ones((np.shape(x)[0], 1))  ##
    X = np.concatenate([ones_matrix, x], 1).T
    # print(np.shape(X))
    # print(np.shape(y))
    time_start = time()
    exact_model = np.dot(np.dot(np.linalg.pinv(np.dot(X, X.T)), X), y)
    time_end = time()
    print(f"[Time] {time_end - time_start} секунд")
    exact_model_result = np.array((exact_model[1][0], exact_model[0][0]))
    if view_img:
        with open(filename_train, 'r') as f:
            data = np.loadtxt(f, delimiter=',')
        # split to initial arrays
        x_train, y_train = np.hsplit(data, 2)
        # print(exact_model_result)
        h = np.dot(exact_model.T, X).T
        # and check if it's ok
        plt.title("Linear regression [exact solution]")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(x_train, y_train, "b.", label='experiment')
        plt.plot(x, h, "r", label='model')
        plt.legend()
        plt.show()
    return exact_model_result


def check(model, ground_truth):
    if len(model) != len(ground_truth):
        print("Model is inconsistent")
        return False
    else:
        r = np.dot(model - ground_truth, model - ground_truth) / (np.dot(ground_truth, ground_truth))
        # print(r)
        if r < 0.01:
            return True
        else:
            return False


# Ex1: make the same with polynoms

# generates x and y numpy arrays for
# y = a_n*X^n + ... + a2*x^2 + a1*x + a0 + noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    if len(a) != (n + 1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0, n + 1):
        y = y + a[i] * np.power(x, i) + noise * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')


def polynomial_regression_numpy(filename):
    print("Ex1: your code here")
    time_start = time()
    print("Ex1: your code here")
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")


# Ex.2 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i (shape is 1 x N - the same as theta)
# x and y are both vectors

def gradient_descent(dJ, alpha, filename_train, filename_test, size_of_batch=0, view_img=0):
    if dJ == "Полный":
        x, y, X, Y = create_batch(filename_test)
    if dJ == "Мини-batch":
        x, y, X, Y = create_batch(filename_test, size_of_batch)
    if dJ == "Стохастический":
        x, y, X, Y = create_batch(filename_test, size_of_batch)

    theta = np.zeros((X.shape[0], 1))  # Начальные значения
    delta_theta = 1
    kr = theta.shape[0] * 0.001 ** theta.shape[0]
    epoch = 1
    epoch_max = 1000
    time_start = time()
    while delta_theta >= kr and epoch < epoch_max:  # Критерий сходимости: Δθj >= 0.001
        if dJ == "Мини-batch":
            x, y, X, Y = create_batch(filename_test, size_of_batch)
        if dJ == "Стохастический":
            x, y, X, Y = create_batch(filename_test, size_of_batch)
        theta, delta_theta = get_dJ(X, Y.T, alpha, theta)
        alpha = (0.1 * epoch_max) / (epoch + epoch_max)
        epoch = epoch + 1
    time_end = time()
    print(f"[Time] {time_end - time_start} секунд")
    theta_result = np.array((theta[1][0], theta[0][0]))

    if view_img:
        x, y, X_train, Y_train = create_batch(filename_train, 80)
        h = np.dot(theta.T, X_train).T
        with open(filename_train, 'r') as f:
            data = np.loadtxt(f, delimiter=',')
        x_train, y_train = np.hsplit(data, 2)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(x_train, y_train, "b.", label='Эксперимент')
        plt.plot(x, h, "r", label='Модель')
        plt.show()
    return theta_result


# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, alpha, theta, lamb=0):
    lamb = 0
    theta_new = np.zeros_like(theta)
    h = np.dot(theta.T, x)
    theta_new = theta - alpha / np.shape(x)[0] * (np.dot(x, (h - y).T) + lamb * theta)  # for j in range(θ_gd.shape[0]):
    delta_theta = np.dot((theta - theta_new).T, (theta - theta_new))[0, 0]
    return theta_new, delta_theta


def get_J(h, y, theta, lamb):
    summ = 0
    reg = 0
    for i in range(np.shape(y)[0]):
        summ = summ + (h[0, i] - y[0, i]) ** 2

    if lamb != 0:
        for j in range(theta.shape[0]):
            reg = reg + theta[j, 0] ** 2

    return (summ + lamb * reg) / (2 * np.shape(y)[0])


def get_polynomial_dJ(x, y, alpha, theta, j_values, lamb):
    theta_new = np.zeros_like(theta)
    h = np.dot(theta.T, x)
    theta_new = theta - alpha / np.shape(x)[0] * (np.dot(x, (h - y).T) + lamb * theta)  # for j in range(θ_gd.shape[0]):
    delta_theta = np.dot((theta - theta_new).T, (theta - theta_new))[0, 0]
    j_values.append(get_J(h, y, theta, lamb))
    return theta_new, delta_theta


def converting_XYdata_to_matrix(i, filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)
    X = x.T;
    Y = np.power(x, 2)
    for i in range(2, i):  # вручную прописано
        X = np.vstack((X, np.power(x.T, i)))
    ones_matrix = np.ones((np.shape(x)[0], 1))
    X = np.vstack((ones_matrix.T, X))
    return X, y


def polynomial_gradient_descent(alpha, lamb, i, filename_test, filename_train, size_of_batch=0, view_img=0):

    X, Y = converting_XYdata_to_matrix(i, filename_train)
    theta = np.zeros((X.shape[0], 1))  # Начальные значения
    delta_theta = 1
    kr = theta.shape[0] * 0.001 ** theta.shape[0]
    epoch = 1
    epoch_max = 10000
    j_values = []
    time_start = time()
    while delta_theta >= kr and epoch < epoch_max:  # Критерий сходимости: Δθj >= 0.001
        theta, delta_theta = get_polynomial_dJ(X, Y.T, alpha, theta, j_values, lamb)
        # alpha = (0.1 * epoch_max) / (epoch + epoch_max)
        epoch = epoch + 1
    time_end = time()
    print(f"[Time] {time_end - time_start} секунд")
    # theta_result = np.array((theta[0][0], theta[1][0], theta[2][0], theta[3][0], theta[4][0]))

    if view_img:
        X_test, Y_test = converting_XYdata_to_matrix(len(array_odds), filename_test)
        h = np.dot(theta.T, X_test).T
        with open(filename_train, 'r') as f:
            data = np.loadtxt(f, delimiter=',')
        x_train, y_train = np.hsplit(data, 2)
        plt.title("Polynomial regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(X[1], Y, "b.", label='experiment')
        plt.plot(X_test[1], h, "r.", label='model')
        plt.legend()
        plt.show()

    return theta


def polynomial_gradient_descent_for_curves(alpha, lamb, X, Y, j_values=[]):


    theta = np.zeros((X.shape[0], 1))  # Начальные значения
    delta_theta = 1
    theta_new = np.zeros_like(theta)
    kr = theta.shape[0] * 0.001 ** theta.shape[0]
    epoch = 1
    epoch_max = 500
    time_start = time()

    while delta_theta >= kr and epoch < epoch_max:  # Критерий сходимости: Δθj >= 0.001
        h = np.dot(theta.T, X)
        theta_new = theta - alpha / np.shape(X)[0] * (
                np.dot(X, (h - Y).T) + lamb * theta)  # for j in range(θ_gd.shape[0]):
        delta_theta = np.dot((theta - theta_new).T, (theta - theta_new))[0, 0]
        theta = theta_new
        j_values.append(get_J(h, Y, theta, lamb))
        epoch = epoch + 1
    time_end = time()
    print(f"[Time] {time_end - time_start} секунд")
    return theta


# try each of gradient decsent (complete, minibatch, sgd) for varius alphas
# L - number of iterations
# plot results as J(i)

def minimize(alpha, lamb, theta, filename_test, filename_train):
    X_train, Y_train = converting_XYdata_to_matrix(len(array_odds), filename_train)
    X_test, Y_test = converting_XYdata_to_matrix(len(array_odds), filename_test)
    for lamb, col in [[0.001, 'r']]:
        X_tr = np.vstack((X_train[0], X_train[1]))
        Y_tr = Y_train.T
        X_t = np.vstack((X_test[0], X_test[1]))
        Y_t = Y_test.T

        J_train = []
        J_test = []
        for n in range(1, 6):
            theta = polynomial_gradient_descent_for_curves(alpha, lamb, X_tr, Y_tr)
            print(theta)
            J_train.append(get_J(np.dot(theta.T, X_tr), Y_tr, theta, lamb))
            J_test.append(get_J(np.dot(theta.T, X_t), Y_t, theta, lamb))
            X_tr = np.vstack((X_tr, np.power(X_tr[1,], n)))
            X_t = np.vstack((X_t, np.power(X_t[1,], n)))

        plt.plot(J_train, "r-", label=f'train λ = {lamb}')
        plt.plot(J_test, "b-", label=f'test λ = {lamb}')

    plt.xlabel("degree")
    plt.ylabel("J(θ)")
    plt.show()
    return


def learning_curves(alpha, lamb, filename_test, filename_train):
    X_train, Y_train = converting_XYdata_to_matrix(5, filename_train)
    X_test, Y_test = converting_XYdata_to_matrix(5, filename_test)
    Y_train = Y_train.T
    Y_test = Y_test.T

    J_train = []
    J_test = []

    for m in range(1, Y_train.shape[1]):
        theta = polynomial_gradient_descent_for_curves(alpha, lamb, X_train[:, :m], Y_train[:, :m])
        if m == 1:
            print(theta)
            print(X_train[:, :m])
            print(np.dot(theta.T, X_train[:, :m]))
            print(Y_train[:, :m])
        J_train.append(get_J(np.dot(theta.T, X_train[:, :m]), Y_train[:, :m], theta, lamb))
        J_test.append(get_J(np.dot(theta.T, X_test), Y_test, theta, lamb))

    plt.title("Learning curves")
    plt.xlabel("size")
    plt.ylabel("J(θ)")
    plt.plot(J_test, 'b', label="test")
    plt.plot(J_train, 'r', label="train")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    generate_linear(1, 3, 1, 'linear.csv', 1000)
    array_odds = [1, -20, 0, 7, -4]
    generate_poly(array_odds, 4, 1, "polynomial.csv", 100)
    train_test_split_To_file('linear.csv', 'linear_train.csv', 'linear_test.csv')
    train_test_split_To_file('polynomial.csv', 'polynomial_train.csv', 'polynomial_test.csv')

    print("Calculate values by polyfit")
    model = linear_regression_numpy("linear_train.csv", "linear_test.csv", 1)
    print(f"Values: a = {model[0]}   b = {model[1]}")
    print(f"Checking the correctness of the model built by polyfit: {check(model, np.array([1, -3]))}")
    print("===============================================================================================")
    print("Calculate values by exact solution through the matrices")
    model_exact = linear_regression_exact("linear_train.csv", "linear_test.csv", 1)
    print(f"Values: a = {model_exact[0]}   b = {model_exact[1]}")
    print(
        f"Checking the correctness of the model built by the exact solution through the matrices: {check(model_exact, np.array([1, -3]))}")
    print("===============================================================================================")
    print("Calculate values by Полный gradient descent")
    model_full_gradient_descent = gradient_descent("Полный", 0.1, "linear_train.csv", "linear_test.csv", 0, 1)
    print(f"Values: a = {model_full_gradient_descent[0]}   b = {model_full_gradient_descent[1]}")
    print(
        f"Checking the correctness of the model built by Полный gradient descent: {check(model_full_gradient_descent, np.array([1, -3]))}")
    print("===============================================================================================")
    print("Calculate values by gradient descent (Мини-batch)")
    model_minibatch_gradient_descent = gradient_descent("Мини-batch", 0.1, "linear_train.csv", "linear_test.csv", 10, 1)
    print(f"Values: a = {model_minibatch_gradient_descent[0]}   b = {model_minibatch_gradient_descent[1]}")
    print(
        f"Checking the correctness of the model built by Мини-batch gradient descent: {check(model_minibatch_gradient_descent, np.array([1, -3]))}")
    print("===============================================================================================")
    print("Calculate values by Стохастический gradient descent")
    model_stochastic_gradient_descent = gradient_descent("Стохастический", 0.1, "linear_train.csv", "linear_test.csv", 1, 1)
    print(f"Values: a = {model_stochastic_gradient_descent[0]}   b = {model_stochastic_gradient_descent[1]}")
    print(
        f"Checking the correctness of the model built by Стохастический gradient descent: {check(model_stochastic_gradient_descent, np.array([1, -3]))}")
    print("===============================================================================================")
    # print("Calculate polynomial regression")
    temp = 5
    polynomial_model = polynomial_gradient_descent(0.01, 0.001, 5, 'polynomial_test.csv',"polynomial_train.csv", 0,1)
    print(
        f"Values: a0 = {polynomial_model[0]}   a1 = {polynomial_model[1]} a2 = {polynomial_model[2]} a3 = {polynomial_model[3]} a4 = {polynomial_model[4]}")
    minimize(0.01, 0.001, polynomial_model, 'polynomial_test.csv', 'polynomial_train.csv')
    learning_curves(0.01, 0.001, 'polynomial_test.csv', 'polynomial_train.csv')
