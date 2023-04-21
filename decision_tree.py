from dask.distributed import Client
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from pandas import DataFrame as df
import seaborn as sns
from sklearn.model_selection import train_test_split
from joblib import Parallel, parallel_backend
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
import graphviz
import numpy as np

def decision_tree(x, y, parallel=False, save=False, ip=None, treetype='DecisionTreeClassifier', param_grid=None,
                  cv=False, cv_num=5, seed=None, criterion=None
                  , max_depth=3, multi_class='auto', verbose=1):
    """
    使用决策树算法进行分类或回归，并返回模型对象。
    Args:
        x: 特征矩阵，二维数组。
        y: 目标向量，一维数组。
        parallel: 是否使用 Dask 进行并行计算，默认为 False。
        save: 是否将训练好的模型保存到文件中，默认为 False。
        ip: Dask 集群的 IP 地址，默认为 None。
        treetype: 决策树的类型，支持 DecisionTreeClassifier 和 DecisionTreeRegressor 两种，默认为 DecisionTreeClassifier。
        param_grid: 网格搜索的参数范围，默认为 None。
        cv: 是否进行交叉验证，默认为 False。
        cv_num: 交叉验证的折数，默认为 5。
        seed: 随机种子，默认为 None。
        criterion: 决策树节点分裂的标准，默认为 gini 和 mse 两种（CART），初始为 None，自动根据 treetype 进行选择。
        max_depth: 决策树的最大深度，默认为 3。
        multi_class: 分类问题的类型，支持 auto、ovr 和 multinomial 三种，默认为 auto。
        verbose: 是否显示详细信息，默认为 1。

    Returns:
        训练好的决策树模型对象。

    Raises:
        ValueError: 当 treetype 的取值不在支持的范围内时，抛出异常。
    """
    if (param_grid is None)&(treetype == 'DecisionTreeClassifier'): #参数网格默认值
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    elif (param_grid is None)&(treetype == 'DecisionTreeRegressor'):
        param_grid = {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    if criterion is None:
        if treetype == 'DecisionTreeClassifier':
            criterion = 'gini'
        elif treetype == 'DecisionTreeRegressor':
            criterion = 'mse'
        else:
            raise ValueError("Unknown tree type '{}'".format(treetype))

    if parallel:# 并行计算
        # 1. 创建一个 Dask 集群
        client = Client(ip)
        print('已经连接到 Dask 集群')
        print('集群信息：', client)

        if cv:#交叉验证
            search = RandomizedSearchCV(eval('tree.{}'.format(treetype))(), param_grid, cv=cv_num, random_state=seed, verbose=verbose)
            with parallel_backend('dask'):
                search.fit(x, y)
            model = search.best_estimator_
            print('模型评分：', model.score(x, y))
        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            with parallel_backend('dask'):
                model = eval('tree.{}'.format(treetype))(random_state=seed, criterion=criterion, max_depth=max_depth, multi_class=multi_class)
                model.fit(X_train, y_train)
                print('模型评分：', model.score(X_test, y_test))
    else:
        if cv:
            search = RandomizedSearchCV(eval('tree.{}'.format(treetype))(), param_grid, cv=cv_num)
            search.fit(x, y)
            model = search.best_estimator_
            print('模型评分：', model.score(x, y))

        else:
            model = eval('tree.{}'.format(treetype))(random_state=seed, criterion=criterion, max_depth=max_depth)
            model.fit(x, y)
            print('模型评分：', model.score(x, y))

    if save:
        joblib.dump(model, 'decision_tree_model.pkl')

    return model

def visual_tree(method = 'tree_plot', model = None, x_name=None, y_name=None, save=False, out_file=None):
    # method: tree_plot, graphviz
    # model: 模型
    # x_name: 特征名称
    # y_name: 标签名称
    # save: 是否保存
    # out_file: 保存文件名 仅在method为graphviz时有效 默认为None 保存为decision_tree.
    if method == 'tree_plot':
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
        tree.plot_tree(model, feature_names=x_name, class_names=y_name, filled=True)
        if save:
            fig.savefig('decision_tree.png')
        plt.show()

    elif method == 'graphviz':
        dot_data = tree.export_graphviz(model,
                                        out_file=out_file,
                                        feature_names=x_name,
                                        class_names=y_name,
                                        filled=True)
        graph = graphviz.Source(dot_data)
        if save:
            graph.render('decision_tree')
        graph.view()
    else:
        print('方法错误，可选方法为：tree_plot, graphviz')


def feature_importance_visual(model, feature_names, save=False):

    # model: 模型
    # feature_names: 特征名称
    # save: 是否保存
    # 从大到小排序


    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # 使用sns绘制条形图
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    sns.set_palette('Set2')
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.asanyarray(feature_names)[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Variable Importance Of Decision Tree')
    if save:
        plt.savefig('feature_importance.png')

    plt.show()




'''
使用示例
'''

if __name__ == '__main__':
    data = load_breast_cancer()

    x = data.data
    y = data.target
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    model = decision_tree(x, y, cv=True,seed=42, parallel=True)


    visual_tree(method='graphviz', model=model,
                x_name=data.feature_names, y_name=data.target_names, save=True)
    feature_importances = df(data.feature_names, columns=['feature_names'])
    feature_importances['feature_importances'] = model.feature_importances_

    feature_importance_visual(model, data.feature_names, save=True)