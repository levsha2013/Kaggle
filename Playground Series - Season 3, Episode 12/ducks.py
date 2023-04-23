import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool, cv

random_state = 42
iters = 1000

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X =['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']
Y = ['target']


"""for i in X:
    print(i)
    plt.scatter(train[i], train['calc'], alpha=0.3)
    plt.scatter(test[i], test['calc'], alpha=0.3)
    plt.show()"""

to_norm = X
mod = MinMaxScaler()
mod.fit(train[to_norm])
train[to_norm] = mod.transform(train[to_norm])
test[to_norm] = mod.transform(test[to_norm])


poly = PolynomialFeatures(3, include_bias=False)        # На 8 log regression дает чуть лучше результат
a = pd.DataFrame(poly.fit_transform(train[to_norm]))
train.index = [i for i in range(len(train))]
a.columns = [str(i) for i in a.columns]
train = pd.concat([train, a], axis=1)

poly = PolynomialFeatures(3, include_bias=False)        # На 8 log regression дает чуть лучше результат
a = pd.DataFrame(poly.fit_transform(test[to_norm]))
test.index = [i for i in range(len(test))]
a.columns = [str(i) for i in a.columns]
test = pd.concat([test, a], axis=1)
X.extend(list(a.columns))

from math import log
def log_numeric(x):
    if x != 0:
        return x/abs(x)*log(abs(x))
    else:
        return 0

for i in X:
    test[i] = test[i].apply(log_numeric)
    train[i] = train[i].apply(log_numeric)


train_1, test_1 = train_test_split(train, test_size=0.2, random_state=random_state)

model = LogisticRegression()
model.fit(train[X], train['target'])
pred = model.predict(test_1[X])
a = confusion_matrix(pred, test_1['target'])
print("LogisticRegression", (a[0,0]+a[1,1])/len(test_1))
test['target'] = model.predict(test[X])
test[['id', 'target']].to_csv('titanic_sub_sklearn.csv', index=False)

params = {'iterations': 5000,
          'verbose': 100,
          'random_seed': random_state,
          'learning_rate': 0.001,
          'eval_metric': 'Logloss',
          'loss_function': 'Logloss',
          }



train_data = Pool(data=train_1[X],
                  label=train_1[Y],)
test_data = Pool(data=test_1[X],
                 label=test_1[Y],)
train_full = Pool(data=train[X],
                  label=train[Y],
                  )

model = CatBoostClassifier(**params)
model.fit(train_data, eval_set=(test_data))
pred = model.predict(test_1[X])
a = confusion_matrix(pred, test_1[Y])
print("LogisticRegression", (a[0,0]+a[1,1])/len(test_1))


cv_data = cv(params=params, pool=train_full, fold_count=5, shuffle=True,
             partition_random_seed=random_state, stratified=False, verbose=False)

name_of_metrix = f"test-{params['eval_metric']}-mean"
best_iter = cv_data[cv_data[name_of_metrix] == cv_data[name_of_metrix].min()]['iterations'].values[0]

# ________________ КРОСС-ВАЛИДАЦИЯ - обучение до лучшей итерации____________________ #
params = {'iterations': best_iter+1,
          'verbose': 100,
          'random_seed': random_state,
          'learning_rate': 0.001,
          'eval_metric': 'Logloss',
          'loss_function': 'Logloss',
          }

model = CatBoostClassifier(**params)
model.fit(train[X], train[Y])
print(model.get_feature_importance(prettified=True))

# ________________ ПРЕДСКАЗАНИЕ ПО СОЗДАННОЙ МОДЕЛИ И СОХРАНЕНИЕ В ФАЙЛ ____________________ #
test['target'] = model.predict(test[X])
test[['id', 'target']].to_csv('duck.csv', index=False)

print("Все")