import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# 定义参数初始化函数
def initialize(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b


# 定义ridge损失函数
def l2_loss(X, y, w, b, alpha):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat - y) ** 2) / num_train + alpha * (np.sum(np.square(w)))
    dw = np.dot(X.T, (y_hat - y)) / num_train + 2 * alpha * w
    db = np.sum((y_hat - y)) / num_train
    return y_hat, loss, dw, db


# 定义训练过程
def ridge_train(X, y, learning_rate=0.01, epochs=300):
    loss_list = []
    w, b = initialize(X.shape[1])
    for i in range(1, epochs):
        y_hat, loss, dw, db = l2_loss(X, y, w, b, 0.1)
        w += -learning_rate * dw
        b += -learning_rate * db
        loss_list.append(loss)

        if i % 100 == 0:
            print('epochs %d loss %f' % (i, loss))
        params = {
            'w': w,
            'b': b
        }
        grads = {
            'dw': dw,
            'db': db
        }
    return loss, loss_list, params, grads


# 定义预测函数
def predict(X, params):
    w = params['w']
    b = params['b']

    y_pred = np.dot(X, w) + b
    return y_pred


if __name__ == '__main__':
    data = np.genfromtxt('example.dat', delimiter=',')
    # 选择特征与标签
    x = data[:, 0:100]
    y = data[:, 100].reshape(-1, 1)
    # 加一列
    X = np.column_stack((np.ones((x.shape[0])), x))

    # 划分训练集与测试集
    X_train, y_train = X[:70], y[:70]
    X_test, y_test = X[70:], y[70:]
    print(X_train.shape, y_train, X_test.shape, y_test.shape)
    # 执行训练示例
    loss, loss_list, params, grads = ridge_train(X_train, y_train, 0.01, 1000)
    y_pred = predict(X_test, params)
    r2_score(y_pred, y_test)

    # 简单绘图
    f = X_test.dot(params['w']) + params['b']
    plt.scatter(range(X_test.shape[0]), y_test)
    plt.plot(f, color='darkorange')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    # 训练过程中的损失下降
    plt.plot(loss_list, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
