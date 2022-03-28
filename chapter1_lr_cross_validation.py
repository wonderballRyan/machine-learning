# 导入sklearn diabetes数据接口
import numpy as np
from sklearn.datasets import load_diabetes
# 导入sklearn打乱数据函数
from sklearn.utils import shuffle

# 获取diabetes数据集
diabetes = load_diabetes()
# 获取输入和标签
data, target = diabetes.data, diabetes.target
# 打乱数据集
X, y = shuffle(data, target, random_state=13)
X = X.astype(np.float32)
data = np.concatenate((X, y.reshape((-1, 1))), axis=1)


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


def k_fold_cross_validation(items, k, randomize=True):
    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        training = np.array(training)
        validation = np.array(validation)
        yield training, validation


for training, validation in k_fold_cross_validation(data, 5):
    X_train = training[:, :10]
    y_train = training[:, -1].reshape((-1, 1))
    X_valid = validation[:, :10]
    y_valid = validation[:, -1].reshape((-1, 1))
    loss5 = []
    loss, params, grads = linear_train(X_train, y_train, 0.001, 100000)
    loss5.append(loss)
    score = np.mean(loss5)
    print('five fold cross validation score is', score)
    y_pred = predict(X_valid, params)
    valid_score = np.sum((y_pred - y_valid) ** 2) / len(X_valid)
    print('valid_score is', valid_score)
