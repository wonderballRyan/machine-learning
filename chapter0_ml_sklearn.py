from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 导入数据
X, y = load_iris(return_X_y=True)
# 拟合模型
clf = LogisticRegression(random_state=0).fit(X, y)
# 预测
print(clf.predict_proba(X[:2, :]))
# 模型准确率
print(clf.score(X, y))
