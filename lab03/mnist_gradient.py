import time
import numpy as np
import matplotlib.pyplot as plt

# Изображения из набора данных имеют размер 28 x 28.
# Они сохраняются в файлах данных csv mnist_train.csv и mnist_test.csv.

image_size = 28
train_data = np.loadtxt("./mnist_train.csv", delimiter=",")
test_data = np.loadtxt("./mnist_test.csv", delimiter=",")


def to_float(image_data):
    fac = 0.99 / 255
    return np.asfarray(image_data) * fac + 0.01


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_func(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1 / m) * (((-y).T @ np.log(h + epsilon)) - ((1 - y).T @ np.log(1 - h + epsilon)))
    return cost


def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost = np.zeros((iterations, 1))

    for i in range(iterations):
        params = params - (learning_rate / m) * (X.T @ (sigmoid(X @ params) - y))
        cost[i] = cost_func(X, y, params)
    return cost, params


def predict(X, params):
    return np.round(sigmoid(X @ params))


def print_number(sample, title="?"):
    digit = np.reshape(sample, (image_size, image_size))
    plt.imshow(digit, cmap="Greys")
    plt.title(title)
    plt.show()


def main():
    # Делим изображения на два класса по чётности цифры
    for dataset in [train_data, test_data]:
        for sample in dataset:
            sample[0] = sample[0] % 2

    train_images = to_float(train_data[:, 1:])
    test_images = to_float(test_data[:, 1:])
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    # Демонстрация данных из MNIST
    for i in range(5):
        img = train_images[i].reshape((image_size, image_size))
        print_number(img, "Number")

    # Параметры
    n = np.size(train_images, 1)
    params = np.zeros((n, 1))
    iterations = 1500
    learning_rate = 0.03

    cost, params_new = gradient_descent(train_images, train_labels, params, learning_rate, iterations)
    y_pred = predict(test_images, params_new)

    # Предсказания для тестовой выборки
    score = float(sum(y_pred == test_labels)) / float(len(test_labels))
    print(score)

    # Демонстрация обработанных данных
    for i in range(10):
        img = test_images[i]
        is_num_even = y_pred[i]
        print(is_num_even)
        print_number(img, "Нечетное" if is_num_even else "Четное")

    # Подсчет времени
    start = time.time()
    predict(test_images, params_new)
    print("Время: ", time.time() - start)


if __name__ == '__main__':
    main()
