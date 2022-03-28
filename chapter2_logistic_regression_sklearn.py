import numpy as np
import matplotlib.pyplot as plt
# 导入生成分类数据函数
from sklearn.datasets._samples_generator import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# 生成100*2的模拟二分类数据集
X, labels = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2
)
# 设置随机数种子
rng = np.random.RandomState(2)
# 对生成的特征数据添加一组均匀分布噪声
X += 2 * rng.uniform(size=X.shape)
# 标签类别数
unique_labels = set(labels)
# 根据标签类别数设置颜色
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# 绘制模拟数据的散点图
for k, col in zip(unique_labels, colors):
    x_k = X[labels == k]
    plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k", markersize=14)
plt.title('Simulated binary data set')
plt.show()
print(X.shape, labels.shape)

labels = labels.reshape((-1, 1))
data = np.concatenate((X, labels), axis=1)
print(data.shape)

# 训练集与测试集的简单划分
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], labels[:offset]
X_test, y_test = X[offset:], labels[offset:]
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

print('X_train=', X_train.shape)
print('X_test=', X_test.shape)
print('y_train=', y_train.shape)
print('y_test=', y_test.shape)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)