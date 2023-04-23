import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, cv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib import

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
random_state = 50

all_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'],
cat_featches = ['Survived', 'Pclass', 'Name', 'Sex', 'Embarked']

print("Все")