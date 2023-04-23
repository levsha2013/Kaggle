from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures, LabelBinarizer
from sklearn.metrics import confusion_matrix
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.decomposition import PCA

random_state = 42
iters = 1000

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


X = test.columns
Y = 'prognosis'

pca = PCA(0.9999)
train_pca = pd.concat([pd.DataFrame(pca.fit_transform(train[X])), train[Y]], axis=1)

train_1, test_1 = train_test_split(train_pca, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(train_1[0], train_1[Y])

print("Все")