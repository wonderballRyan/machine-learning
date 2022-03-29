# 导入线性模型模块
from sklearn import linear_model
import numpy as np

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

# 创建lasso模型实例
sk_lasso = linear_model.Lasso(alpha=0.1)
# 对训练集进行拟合
sk_lasso.fit(X_train, y_train)
# 打印模型相关系数
print("sklearn Lasso intercept :", sk_lasso.intercept_)
print("\nsklearn Lasso coefficients :\n", sk_lasso.coef_)
print("\nsklearn Lasso number of iterations :", sk_lasso.n_iter_)
