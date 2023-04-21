import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import joblib
from joblib import Parallel, parallel_backend
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import Client

def random_forest(x, y, parallel=False, save=False, ip=None,
                  cv=False, param_grid=None, verbose=1, cv_num=5,
                  Foresttype='RandomForestClassifier', seed=None, criterion=None, max_depth=3, n_estimators=100, test_size=0.2,
                  min_samples_split=2, min_samples_leaf=1, bootstrap=True, n_jobs=None):
    '''
    使用随机森林算法进行分类或回归，并返回模型对象。
    :param x: ndarray 特征矩阵。
    :param y: ndarray 目标向量。
    :param parallel: bool 是否使用 Dask 进行并行计算，默认为 False。
    :param save: bool 是否将训练好的模型保存到文件中，默认为 False。
    :param ip: ip 地址，用于连接 Dask 集群。
    :param cv: bool 是否使用交叉验证，默认为 False。
    :param param_grid: dict 参数网格, 仅当 cv=True 时生效。
    :param verbose: int 交叉验证时的详细程度，仅当 cv=True 时生效。
    :param cv_num: int 交叉验证的折数，仅当 cv=True 时生效。
    :param Foresttype: str 随机森林类型，可选值为 'RandomForestClassifier' 或 'RandomForestRegressor'。
    :param seed: int 随机种子。
    :param criterion: str 分类随机森林的划分标准，当为分类随机森林时可选值为 'gini' 或 'entropy'，为回归随机森林时可选值为 'mse' 或 'mae'。
    :param max_depth: int 随机森林的最大深度。
    :param n_estimators: int 随机森林的树的数量。
    :param test_size: float 测试集的比例。
    :param min_samples_split: int 内部节点再划分所需最小样本数。
    :param min_samples_leaf: int 叶子节点最少样本数。
    :param bootstrap: bool 是否有放回的抽样。
    :return: model 模型对象。
    '''

    if cv:
        if (param_grid is None) & (Foresttype == 'RandomForestClassifier'):  # 分类随机森林参数网格默认值
            param_grid = {
                'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            }
        elif (param_grid is None) & (Foresttype == 'RandomForestRegressor'):  # 回归随机森林参数网格默认值
            param_grid = {
                'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'criterion': ['squared_error', 'absolute_error']
            }
        else: # 用户自定义参数网格
            pass

    if criterion is None:#根据随机森林的类型设定划分标准
        if Foresttype == 'RandomForestClassifier':
            criterion = 'gini'
        elif Foresttype == 'RandomForestRegressor':
            criterion = 'squared_error'
        else:
            raise ValueError('Foresttype 只能为 RandomForestClassifier 或 RandomForestRegressor')

    if parallel:# 并行计算
        # 1. 创建一个 Dask 集群
        client = Client(ip)
        print('已经连接到 Dask 集群')
        print('集群信息：', client)

        if cv:#交叉验证
            search = RandomizedSearchCV(eval('sklearn.ensemble.{}'.format(Foresttype))(), param_grid, cv=cv_num, random_state=seed, verbose=verbose)
            with parallel_backend('dask'):
                search.fit(x, y)
            model = search.best_estimator_
            print('模型评分：', model.score(x, y))
        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
            with parallel_backend('dask'):
                model = eval('sklearn.ensemble.{}'.format(Foresttype))(random_state=seed, criterion=criterion, max_depth=max_depth,
                                                                       n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                       min_samples_leaf=min_samples_leaf,
                                                                       bootstrap=bootstrap, n_jobs=n_jobs)
                model.fit(X_train, y_train)
                print('模型评分：', model.score(X_test, y_test))
    else:
        if cv:
            search = RandomizedSearchCV(eval('sklearn.ensemble.{}'.format(Foresttype))(), param_grid, cv=cv_num)
            search.fit(x, y)
            model = search.best_estimator_
            print('模型评分：', model.score(x, y))

        else:
            model = eval('sklearn.ensemble.{}'.format(Foresttype))(random_state=seed, criterion=criterion,
                                                                   max_depth=max_depth,
                                                                   n_estimators=n_estimators,
                                                                   min_samples_split=min_samples_split,
                                                                   min_samples_leaf=min_samples_leaf,
                                                                   bootstrap=bootstrap, n_jobs=n_jobs)
            model.fit(x, y)
            print('模型评分：', model.score(x, y))

    if save:#保存模型
        joblib.dump(model, 'random_forest_{}.pkl'.format(Foresttype))
    return model

def visual_randomforest_importance (model, feature_names, Foresttype='RandomForestClassifier', save=False):
    '''
    可视化随机森林的特征重要性
    :param model:
    :param feature_names:
    :param Foresttype:
    :param save:
    :return:
    '''
    if Foresttype == 'RandomForestClassifier':
        importance = model.feature_importances_
    elif Foresttype == 'RandomForestRegressor':
        importance = model.feature_importances_
    else:
        raise ValueError('Foresttype 只能为 RandomForestClassifier 或 RandomForestRegressor')

    importance = pd.DataFrame(importance, index=feature_names, columns=['importance'])
    importance = importance.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 10))
    sns.barplot(x=importance['importance'], y=importance.index, orient='h')
    plt.title('Feature Importance of Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    if save:
        plt.savefig('random_forest_importance.png')
    plt.show()


#测试
if __name__ == '__main__':
    x, y = load_breast_cancer(return_X_y=True)
    # 随机森林分类
    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error']
    }
    model = random_forest(x, y, Foresttype='RandomForestClassifier', parallel=False,
                          cv=False, verbose=5, param_grid=param_grid, save=True)

    # 可视化特征重要性
    visual_randomforest_importance(model, load_breast_cancer().feature_names, Foresttype='RandomForestClassifier', save=True)