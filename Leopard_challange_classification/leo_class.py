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
"""Данные не одинаковые совсем!!! на трейне 80%, на тесте 0,31 подобно одинаковому - это жесть"""
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


X = ['ID', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
       'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic',
       'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
       'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 'AST',
       'ALT', 'Gtp', 'oral', 'dental caries', 'tartar']
Y = ['smoking']
cat_featches = ['oral', 'dental caries', 'tartar']


train_1, test_1 = train_test_split(train, test_size=0.2, random_state=random_state)


params = {'iterations': 500,
          'verbose': 100,
          'random_seed': random_state,
          'learning_rate': 0.05,
          'eval_metric': 'Logloss',
          'loss_function': 'Logloss',
          }

train_data = Pool(data=train_1[X],
                  label=train_1[Y],
                  cat_features=cat_featches)
test_data = Pool(data=test_1[X],
                 label=test_1[Y],
                 cat_features=cat_featches)
train_full = Pool(data=train[X],
                  label=train[Y],
                  cat_features=cat_featches
                  )

model = CatBoostClassifier(**params)
model.fit(train_data, eval_set=(test_data))
pred = model.predict(test_1[X])
a = confusion_matrix(pred, test_1[Y])
print("CatBoostClassifier", (a[0,0]+a[1,1])/len(test_1))

cv_data = cv(params=params, pool=train_full, fold_count=5, shuffle=True,
             partition_random_seed=random_state, stratified=False, verbose=False)

name_of_metrix = f"test-{params['eval_metric']}-mean"
best_iter = cv_data[cv_data[name_of_metrix] == cv_data[name_of_metrix].min()]['iterations'].values[0]

# ________________ КРОСС-ВАЛИДАЦИЯ - обучение до лучшей итерации____________________ #
params = {'iterations': best_iter+1,
          'verbose': 100,
          'random_seed': random_state,
          'learning_rate': 0.05,
          'eval_metric': 'Logloss',
          'loss_function': 'Logloss',
          'cat_features': cat_featches
          }

model = CatBoostClassifier(**params)
model.fit(train[X], train[Y])
print(model.get_feature_importance(prettified=True))

# ________________ ПРЕДСКАЗАНИЕ ПО СОЗДАННОЙ МОДЕЛИ И СОХРАНЕНИЕ В ФАЙЛ ____________________ #
test['smoking'] = model.predict_proba(test[X])
test[['ID', 'smoking']].to_csv('leo_class.csv', index=False)
print("Все")