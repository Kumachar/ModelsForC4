from dask.distributed import Client
from joblib import Parallel, parallel_backend
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from dask_ml.model_selection import RandomizedSearchCV
import time

client = Client('tcp://192.168.1.196:8786')  # create local cluster
digits = load_digits()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model1 = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(model1, param_grid, cv=5)

t0 = time.time()
print("Running grid search with dask backend start")


with parallel_backend('dask'):
    grid_search.fit(digits.data, digits.target)


print("Running grid search with dask backend end")
t1 = time.time()
print("Time taken: ", t1 - t0)
