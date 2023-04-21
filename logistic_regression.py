from dask.distributed import Client
import joblib
import dask
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from joblib import Parallel, parallel_backend
from sklearn.linear_model import LogisticRegression
from dask_ml.model_selection import RandomizedSearchCV

def logistic_regression(x, y, parallel=False, save=False, ip=None, penalty='l2', solver='lbfgs', max_iter= 1000, C=1.0, multi_class='auto', n_jobs=-1):

    if parallel:
        client = Client(ip)
        print('已经连接到 Dask 集群')
        print('集群信息：', client)

        with parallel_backend('dask'):
            model = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter, C=C, multi_class=multi_class, n_jobs= n_jobs)
            model.fit(x, y)
    else:
        model = LogisticRegression()
        model.fit(x, y)
    if save:
        joblib.dump(model, 'logistic_regression_model.pkl')
    client.close()
    return model

if __name__ == '__main__':
    data = load_breast_cancer()
    x = data.data
    y = data.target
    model = logistic_regression(x, y, save=True, parallel=True, ip=None)
    print(model.score(x, y))