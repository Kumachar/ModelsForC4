from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, parallel_backend
import seaborn as sns
from dask.distributed import Client
import numpy as np

def SVM(x, y, parallel=False, save=False, ip=None, seed=None, test_size=0.2,
                  cv=False, param_grid=None, verbose=1, cv_num=5,
        gamma='auto', random_state=42, kernel='rbf', C=1.0, degree=3, coef0=0.0, shrinking=True,
        probability=False, tol=0.001, cache_size=200, class_weight=None, verbosesvm=False, max_iter=-1
        ):
    '''
    :param x: 训练数据
    :param y: 训练标签
    :param parallel: 是否并行计算
    :param save: 是否保存模型
    :param ip: Dask 集群的 IP 地址
    :param seed: 随机种子
    :param test_size: 测试集大小
    :param cv: 是否进行交叉验证
    :param param_grid: 交叉验证参数
    :param verbose: 交叉验证详细程度
    :param cv_num: 交叉验证折数
    :param gamma: 核函数系数
    :param random_state: 随机种子
    :param kernel: 核函数
    :param C: 惩罚系数
    :param degree: 多项式核函数的阶数
    :param coef0: 核函数中的独立项
    :param shrinking: 是否使用收缩启发式
    :param probability: 是否启用概率估计
    :param tol: 停止训练的误差值大小
    :param cache_size: 缓存大小
    :param class_weight: 类别权重
    :param verbosesvm: 是否输出训练过程
    :param max_iter: 最大迭代次数
    :return: 模型
    '''

    if param_grid is None:
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

    if parallel:# 并行计算
        # 1. 创建一个 Dask 集群
        client = Client(ip)
        print('已经连接到 Dask 集群')
        print('集群信息：', client)

        if cv:#交叉验证
            search = RandomizedSearchCV(SVC(), param_grid, cv=cv_num, random_state=seed, verbose=verbose)
            with parallel_backend('dask'):
                search.fit(x, y)
            model = search.best_estimator_
            print('模型评分：', model.score(x, y))
        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
            with parallel_backend('dask'):
                model = SVC(gamma=gamma, random_state=random_state, kernel=kernel, C=C, degree=degree, coef0=coef0,
                            shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                            verbose=verbosesvm, max_iter=max_iter)
                model.fit(X_train, y_train)
                print('模型评分：', model.score(X_test, y_test))
    else:
        if cv:
            search = RandomizedSearchCV(SVC(), param_grid, cv=cv_num, random_state=seed, verbose=verbose)
            search.fit(x, y)
            model = search.best_estimator_
            print('模型评分：', model.score(x, y))

        else:
            model = SVC(gamma=gamma, random_state=random_state, kernel=kernel, C=C, degree=degree, coef0=coef0,
                            shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                            verbose=verbosesvm, max_iter=max_iter)
            model.fit(x, y)
            print('模型评分：', model.score(x, y))

    if save:#保存模型
        joblib.dump(model, 'SVM.pkl')
    return model

def visual_svm_simple(model, x, y, y_name, save=False):
    '''
    :param model: 模型
    :param x: 训练数据
    :param y: 训练标签
    :param save: 是否保存图像
    :return:
    '''
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, style=y)
    sns.scatterplot(x=model.support_vectors_[:, 0], y=model.support_vectors_[:, 1], color='red', s=100)
    plt.title('SVM')
    #绘制分类结果
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    x_test = np.c_[x1.ravel(), x2.ravel()]
    y_predict = model.predict(x_test).reshape(x1.shape)
    plt.contourf(x1, x2, y_predict, alpha=0.3)

    #legend
    plt.legend(y_name, loc='upper right')
    if save:
        plt.savefig('svm.png')
    plt.show()

def visual_svm_fine(model, x, y, y_name, save=False):
    '''
    :param model: 模型
    :param x: 训练数据
    :param y: 训练标签
    :param save: 是否保存图像
    :return:
    '''
    sns.set_style('darkgrid')
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, style=y, palette='Set1')
    sns.scatterplot(x=model.support_vectors_[:, 0], y=model.support_vectors_[:, 1], color='red', s=100, alpha=0.5)
    plt.title('SVM')
    #绘制分类结果
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 1000), np.linspace(x2_min, x2_max, 1000))
    x_test = np.c_[x1.ravel(), x2.ravel()]
    y_predict = model.predict(x_test).reshape(x1.shape)
    plt.contourf(x1, x2, y_predict, alpha=0.3, cmap='coolwarm')

    #legend
    plt.legend(y_name, loc='upper right')
    if save:
        plt.savefig('svm.png')
    plt.show()


if __name__ == '__main__':
    iris = load_iris()
    x = iris.data[:,:2]
    y = iris.target
    model = SVM(x, y, cv=False, verbose=5)
    visual_svm_simple(model, x, y, iris.target_names, save=True)
    print(model)
