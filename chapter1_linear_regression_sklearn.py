from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import KFold


# 交叉验证
def cross_validate(model, x, y, folds=5, repeats=5):
    ypred = np.zeros((len(y), repeats))
    score = np.zeros(repeats)
    for r in range(repeats):
        i = 0
        print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
        x, y = shuffle(x, y, random_state=r)  # shuffle data before each repeat
        kf = KFold(n_splits=folds, random_state=i + 1000)  # random split, different each time
        for train_ind, test_ind in kf.split(x):
            print('Fold', i + 1, 'out of', folds)
            xtrain, ytrain = x[train_ind, :], y[train_ind]
            xtest, ytest = x[test_ind, :], y[test_ind]
            model.fit(xtrain, ytrain)
            # print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
            ypred[test_ind] = model.predict(xtest)
            i += 1
        score[r] = r2_score(ypred[:, r], y)
    print('\nOverall R2:', str(score))
    print('Mean:', str(np.mean(score)))
    print('Deviation:', str(np.std(score)))
    pass


if __name__ == '__main__':
    diabetes = load_diabetes()
    data = diabetes.data
    target = diabetes.target
    X, y = shuffle(data, target, random_state=13)
    X = X.astype(np.float32)
    y = y.reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients:\n', regr.coef_)
    # The mean squared error
    print('Mean squared error:%.2f' % r2_score(y_test, y_pred))
    # Explained variance score:1 is perfect prediction
    print('Variance score:.2f' % r2_score(y_test, y_pred))
    print(r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(range(X_test.shape[0]), y_test, color='red')
    plt.plot(range(X_test.shape[0]), y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    cross_validate(regr, X, y, folds=5, repeats=5)
