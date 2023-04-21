from random_forest import random_forest
from sklearn.datasets import load_iris
from random_forest import visual_randomforest_importance

iris = load_iris()
x = iris.data
y = iris.target
model = random_forest(x, y, parallel=True, Foresttype='RandomForestClassifier', cv=True, verbose=5)
visual_randomforest_importance(model, x.columns, Foresttype='RandomForestClassifier', save=True)



