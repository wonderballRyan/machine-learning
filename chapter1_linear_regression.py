import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# 初始化模型参数
def initialize_params(dims):
    """
    输入：
    dims:训练数据变量维度
    输出：
    w:初始化权重参数值
    b:初始化偏差参数值
    """
    # 初始化权重参数为零矩阵
    w = np.zeros((dims, 1))
    # 初始化偏差参数为0
    b = 0
    return w, b


# 定义模型主体
# 包括线性回归公式、均方损失和参数偏导三部分
def linear_loss(X, y, w, b):
    """
    输入：
    :param X: 输入变量矩阵
    :param y: 输出标签向量
    :param w: 变量参数权重矩阵
    :param b: 偏差项
    输出:
    y_hat:线性模型预测输出
    loss:均方损失值
    dw:权重参数一阶偏导
    db:偏差项一阶偏导
    """
    # 训练样本数量
    num_train = X.shape[0]
    # 训练样本特征数量
    num_feature = X.shape[1]
    # 线性回归预测输出
    y_hat = np.dot(X, w) + b
    # 计算预测输出与实际标签之间的均方损失
    loss = np.sum((y_hat - y) ** 2) / num_train
    # 基于均方损失对权重参数的一阶偏导数
    dw = np.dot(X.T, (y_hat - y)) / num_train
    # 基于均方损失对偏差项的一阶偏导数
    db = np.sum((y_hat - y)) / num_train
    return y_hat, loss, dw, db


# 定义线性回归模型
def linear_train(X, y, lr=0.01, epochs=10000):
    """
    输入：
    :param X:输入变量矩阵
    :param y: 输出标签向量
    :param lr: 学习率
    :param epochs: 训练迭代次数
    输出:
    loss_his:每次迭代的均方损失
    params:优化后的参数字典
    grads:优化后的参数梯度字典
    """
    # 记录训练损失的空列表
    loss_his = []
    # 初始化模型参数
    w, b = initialize_params(X.shape[1])
    # 迭代训练
    for i in range(1, epochs):
        # 计算当前迭代的预测值、损失和梯度
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        # 基于梯度下降得参数更新
        w += -lr * dw
        b += -lr * db
        # 记录当前迭代得损失
        loss_his.append(loss)
        # 每10000次迭代打印当前损失信息
        if i % 10000 == 0:
            print("epoch %d loss %f" % (i, loss))
        # 将当前迭代步优化后的参数保存到字典
        params = {
            'w': w,
            'b': b
        }
        # 将当前迭代步的梯度保存到字典
        grads = {
            'dw': dw,
            'db': db
        }
    return loss_his, params, grads


# 定义线性回归预测模型
def predict(X, params):
    """
    input:
    :param X: 测试数据集
    :param params: 模型训练参数
    output:
    y_pred:模型预测结果
    """
    # 获取模型参数
    w = params['w']
    b = params['b']
    # 预测
    y_pred = np.dot(X, w) + b
    return y_pred


# 定义R2系数函数
def r2_score(y_test, y_pred):
    """
    input:
    :param y_test: 测试集标签值
    :param y_pred: 测试集预测值
    output:
    r2:R2系数
    """
    # 测试标签值
    y_avg = np.mean(y_test)
    # 总离差平方和
    ss_tot = np.sum((y_test - y_avg) ** 2)
    # 残差平方和
    ss_res = np.sum((y_test - y_pred) ** 2)
    # R2计算
    r2 = 1 - (ss_res / ss_tot)
    return r2


if __name__ == "__main__":
    # 获取diabetes数据集
    diabetes = load_diabetes()
    # 获取输入和标签
    data = diabetes.data
    target = diabetes.target
    # 打乱数据集
    X, y = shuffle(data, target, random_state=13)
    # 按照8/2划分训练集和测试集
    offset = int(X.shape[0] * 0.8)
    # 训练集
    X_train, y_train = X[:offset], y[:offset]
    # 测试集
    X_test, y_test = X[offset:], y[offset:]
    # 将训练集改为列向量的形式
    y_train = y_train.reshape((-1, 1))
    # 将验证集改为列向量的形式
    y_test = y_test.reshape((-1, 1))
    print('X_train‘s shape', X_train.shape)
    print("X_test's shape", X_test.shape)
    print("y_train's shape", y_train.shape)
    print("y_test's shape", y_test.shape)

    # 线性回归模型训练
    loss_his, params, grads = linear_train(X_train, y_train, 0.01, 200000)
    # 打印训练后得到模型参数
    print(params)

    # 基于测试集的预测
    y_pred = predict(X_test, params)
    print('y_pred', y_pred[:5])
    print('y_test', y_test[:5])

    # 计算R2系数
    print(r2_score(y_test, y_pred))

    # 绘图预测图
    f = X_test.dot(params['w']) + params['b']
    plt.scatter(range(X_test.shape[0]), y_test)
    plt.plot(f, color='darkorange')
    plt.xlabel('X_test')
    plt.ylabel('y_test')
    plt.show()

    # 绘制loss图
    plt.plot(loss_his, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
