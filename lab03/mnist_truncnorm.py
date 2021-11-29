import time
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

# Изображения из набора данных имеют размер 28 x 28.
# Они сохраняются в файлах данных csv mnist_train.csv и mnist_test.csv.

image_size = 28
image_pixels = image_size * image_size
train_data = np.loadtxt("./mnist_folder/mnist_train.csv", delimiter=",")
test_data = np.loadtxt("./mnist_folder/mnist_test.csv", delimiter=",")


def to_float(image_data):
    fac = 0.99 / 255
    return np.asfarray(image_data) * fac + 0.01


def print_number(sample, title="?"):
    digit = np.reshape(sample, (image_size, image_size))
    plt.imshow(digit, cmap="Greys")
    plt.title(title)
    plt.show()


def even(num):
    if num % 2 == 0:
        str_even = "Четное"
    else:
        str_even = "Нечетное"
    return  str_even


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


class ClassNeuralNetwork:
    def __init__(self, n_inputs, n_outputs, n_hidden, learning_rate):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.n_inputs)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wih = X.rvs((self.n_hidden, self.n_inputs))
        rad = 1 / np.sqrt(self.n_hidden)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.n_outputs, self.n_hidden))

    def train(self, input_vector, target_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)
        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.who += tmp

        hidden_errors = np.dot(self.who.T, output_errors)
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
        return output_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


def main():
    # 0,01 и 0,99 будут лучше, чем 0 и 1 (одно горячее представление)
    lr = np.arange(10)
    train_images = to_float(train_data[:, 1:])
    test_images = to_float(test_data[:, 1:])
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])
    train_labels_oh = (lr == train_labels).astype(np.float)
    test_labels_oh = (lr == test_labels).astype(np.float)
    train_labels_oh[train_labels_oh == 0] = 0.01
    train_labels_oh[train_labels_oh == 1] = 0.99
    test_labels_oh[test_labels_oh == 0] = 0.01
    test_labels_oh[test_labels_oh == 1] = 0.99

    # Демонстрация данных из MNIST
    # for i in range(5):
    #     img = train_images[i].reshape((image_size, image_size))
    #     print_number(img, "Number")

    # Обучение
    CNN = ClassNeuralNetwork(n_inputs=image_pixels, n_outputs=10, n_hidden=100, learning_rate=0.1)

    for i in range(len(train_images)):
        CNN.train(train_images[i], train_labels_oh[i])

    # Демонстрация обработанных данных
    for i in range(10):
        img = test_images[i]
        res = CNN.run(img)
        is_num_even = even(np.argmax(res))
        print_number(img, is_num_even)

    # Предсказания для тестовой выборки (точность)
    corrects, wrongs = CNN.evaluate(train_images, train_labels)
    print("Точность для тренировочной выборки: ", corrects / (corrects + wrongs))
    corrects, wrongs = CNN.evaluate(test_images, test_labels)
    print("Точность для тестовой выборки: ", corrects / (corrects + wrongs))

    # Подсчет времени
    start = time.time()
    CNN.evaluate(test_images, test_labels)
    print("Время: ", time.time() - start)


if __name__ == '__main__':
    main()
