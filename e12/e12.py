
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


random_state = 42
iters = 1000

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = test.columns.drop('id')
Y = 'target'

grid = GridSearchCV(SVC(), {})

print("Все")