from dask.distributed import Client
import joblib
from joblib import Parallel, parallel_backend
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
from dask_ml.model_selection import RandomizedSearchCV

def linear_regression(x, y, parallel=False, save=False, ip=None):
    if parallel:
        client = Client(ip)

        with parallel_backend('dask'):
            model = LinearRegression()
            model.fit(x, y)
    else:
        model = LinearRegression()
        model.fit(x, y)
    if save:
        joblib.dump(model, 'logistic_regression.pkl')
    return model


'''
if __name__ == '__main__':
    data = load_digits()
    x = data.data
    y = data.target
    model = linear_regression(x, y, save=True, parallel=True, ip=None)
    print(model.score(x, y))
'''


