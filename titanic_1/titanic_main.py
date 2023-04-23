from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv

random_state = 42
iters = 1000

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

X = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
Y = "Survived"
cat_feathures = ['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Embarked']

# ___________________ ПРЕОБРАЗОВАНИЕ ПАРАМЕТРОВ ___________________________ #
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)        # кодирование Sex у train
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)          # кодирование Sex у test
train = train.drop(train[train['Embarked'].isna()].index)                   # выбрасываем 2 train с пустыми Embarked

# заполнение возраста средним по train
age_mean = train['Age'].mean()
train[train['Age'].isna()] = train[train['Age'].isna()].fillna(age_mean)
test[test['Age'].isna()] = test[test['Age'].isna()].fillna(age_mean)
test.loc[test['Fare'].isna(), 'Fare'] = train['Fare'].mean()
# заполнение Cabin no_data
train[train['Cabin'].isna()] = train[train['Cabin'].isna()].fillna('no_data')
train['Cabin'] = train['Cabin'].apply(lambda x: str(x))
test[test['Cabin'].isna()] = test[test['Cabin'].isna()].fillna('no_data')
test['Cabin'] = test['Cabin'].apply(lambda x: str(x))

# разделение имени на парочки
train['Name_1'] = train['Name'].apply(lambda x: x.split(",")[0])
train['Name_2'] = train['Name'].apply(lambda x: x.split(",")[1])
test['Name_1'] = test['Name'].apply(lambda x: x.split(",")[0])
test['Name_2'] = test['Name'].apply(lambda x: x.split(",")[1])

# нормализация MinMax для фитчей Age и Fare
to_norm = ['Age']
mod = MinMaxScaler()
mod.fit(train[to_norm])
train[to_norm] = mod.transform(train[to_norm])
test[to_norm] = mod.transform(test[to_norm])

# полиномизация
poly = PolynomialFeatures(8, include_bias=False)        # На 8 log regression дает чуть лучше результат
a = pd.DataFrame(poly.fit_transform(train[to_norm]))
train.index = [i for i in range(len(train))]
a.columns = [str(i) for i in a.columns]
train = pd.concat([train, a], axis=1)

poly = PolynomialFeatures(8, include_bias=False)        # На 8 log regression дает чуть лучше результат
a = pd.DataFrame(poly.fit_transform(test[to_norm]))
test.index = [i for i in range(len(test))]
a.columns = [str(i) for i in a.columns]
test = pd.concat([test, a], axis=1)


# _________________________ ВИЗУАЛИЗАЦИЯ ПАРАМЕТРОВ ____________________ #
"""for i in numeric_f_ok:
    #plt.scatter(train[i], train[Y])
    print(i)
    train[i].hist()
    plt.show()"""
train.dropna()

# _________________________ УСТАНОВКА ПАРАМЕТРОВ МОДЕЛИ И ОБУЧЕНИЕ SKLEARN ____________________ #
numeric_f_ok = ['Pclass', 'Age', 'Sex']     # лучше - до 80.15% точности
numeric_f_ok.extend(list(a.columns))
train_1, test_1 = train_test_split(train, test_size=0.3, random_state=random_state)    # результат зависит от test_size

model = LogisticRegression(max_iter=1000, random_state=random_state)
model.fit(train_1[numeric_f_ok], train_1[Y])
pred = model.predict(test_1[numeric_f_ok])
a = confusion_matrix(pred, test_1[Y])
print("LogisticRegression", (a[0,0]+a[1,1])/len(test_1))

# создание csv файла
model_2 = LogisticRegression(max_iter=1000, random_state=random_state)
model_2.fit(train[numeric_f_ok], train[Y])
#test['Survived'] = model_2.predict(test[numeric_f_ok])
#test[['PassengerId', 'Survived']].to_csv('titanic_sub_sklearn.csv', index=False)"""

# вариант catboost
numeric_f_ok = ['Pclass', 'Age', 'Sex', 'Name_1', 'Name_2',  'Embarked', 'SibSp',  'Parch', 'Cabin', 'Ticket',  'Fare']
cat_featches = ['Pclass',  'Embarked',  'Name_1', 'Name_2', 'Cabin', 'Ticket']

params_1 = {"verbose": 100,
            'iterations': 1000,
            'cat_features': cat_featches,
            'learning_rate': 0.01,
            'eval_metric': 'Logloss',
          'loss_function': 'Logloss',
            }

model = CatBoostClassifier(**params_1)
model.fit(train_1[numeric_f_ok], train_1[Y], eval_set=(test_1[numeric_f_ok], test_1[Y]))
pred = model.predict(test_1[numeric_f_ok])
a = confusion_matrix(pred, test_1[Y])
print("CatBoostClassifier", (a[0,0]+a[1,1])/len(test_1))


params = {'iterations': 5000,
          'verbose': 100,
          'eval_metric': 'Logloss',
          'loss_function': 'Logloss',
          'random_seed': random_state,
          'learning_rate': 0.002
          }
x_x = ['Pclass', 'Age', 'Sex', 'Name_1', 'Name_2',  'Embarked', 'SibSp',  'Parch', 'Fare']
cat_featches = ['Pclass',  'Embarked',  'Name_1', 'Name_2']
Y = ["Survived"]

train_1, test_1 = train_test_split(train, test_size=0.3)

train_data = Pool(data=train_1[x_x],
                  label=train_1[Y],
                  cat_features=cat_featches)
test_data = Pool(data=test_1[x_x],
                 label=test_1[Y],
                 cat_features=cat_featches)
train_full = Pool(data=train[x_x],
                  label=train[Y],
                  cat_features=cat_featches)
cv_data = cv(params=params, pool=train_data, fold_count=5, shuffle=True,
             partition_random_seed=random_state, stratified=False, verbose=False)
name_of_metrix = f"test-{params['eval_metric']}-mean"
best_iter = cv_data[cv_data[name_of_metrix] == cv_data[name_of_metrix].min()]['iterations'].values[0]

# ________________ КРОСС-ВАЛИДАЦИЯ - обучение до лучшей итерации____________________ #
params = {'iterations': best_iter+1,
          'verbose': 100,
          'random_seed': random_state,
          'learning_rate': 0.002,
          'cat_features': cat_featches,
          'eval_metric': 'Logloss',
          'loss_function': 'Logloss',
          }

#params['iterations'] = best_iter
model = CatBoostClassifier(**params)
model.fit(train[x_x], train[Y])
print(model.get_feature_importance(prettified=True))

# ________________ ПРЕДСКАЗАНИЕ ПО СОЗДАННОЙ МОДЕЛИ И СОХРАНЕНИЕ В ФАЙЛ ____________________ #
test['Survived'] = model.predict(test[x_x])
test[['PassengerId', 'Survived']].to_csv('titanic_sub.csv', index=False)

print("Все")