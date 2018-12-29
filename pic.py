import struct
import numpy as np
import matplotlib.pyplot as plt
import PCA
import math


def read_label():
    path = "train-labels.idx1-ubyte"
    f = open(path, 'rb')
    buf = f.read()
    labels = []
    offset = 0
    offset += struct.calcsize('>ii')
    for i in range(10000):
        label = struct.unpack_from('>B', buf, offset)[0]
        labels.append(label)
        offset += struct.calcsize('>B')
    return labels


def read_pic():
    labels = read_label()
    path = "train-images.idx3-ubyte"
    f1 = open(path, 'rb')
    buf = f1.read()
    data = []
    image_index = 0
    image_index += struct.calcsize('>IIII')
    for i in range(10000):
        temp = struct.unpack_from('>784B', buf, image_index)
        if (labels[i] == 5):
            data.append(list(temp))
        image_index += struct.calcsize('>784B')
    return np.mat(data)


def SNR(X, X2):
    rates = []
    for i in range(len(X)):
        p1 = np.array(X[i])[0]
        p2 = np.array(X2[i])[0]
        val1 = sum(p1 ** 2)
        val2 = sum((p1-p2) ** 2)
        rate = 10 * math.log(val1/val2 , 10)
        rates.append(rate)
    for i in range(10):
        print(rates[i])
    return sum(rates) / len(rates)


def show_pic(x):
    v = np.reshape(x, (28, 28))
    plt.imshow(v, 'gray')
    plt.show()


if __name__ == '__main__':
    X = read_pic()
    Z, X2 = PCA.PCA(X, 100)
    SNR_val = SNR(X, X2)
    show_pic(X[0])
    show_pic(X2[0])
