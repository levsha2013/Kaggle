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

# заполнение Cabin no_data
train[train['Cabin'].isna()] = train[train['Cabin'].isna()].fillna('no_data')
#train['Cabin'] = train['Cabin'].apply(lambda x: str(x))
test[test['Cabin'].isna()] = test[test['Cabin'].isna()].fillna('no_data')
#test['Cabin'] = test['Cabin'].apply(lambda x: str(x))

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

train.dropna()

cat_featches = ['Pclass',  'Embarked',  'Name_1', 'Name_2']

# ________________ КРОСС-ВАЛИДАЦИЯ - обучение до лучшей итерации____________________ #
params = {'iterations': 300,
          'verbose': 100,
          'random_seed': random_state,
          'learning_rate': 0.1,
          'cat_features': cat_featches,
          }

#params['iterations'] = best_iter
model = CatBoostClassifier(**params)
model.fit(train[['Pclass', 'Age', 'Sex',  'Name_1', 'Name_2',  'SibSp',  'Parch', 'Embarked',]],
          train['Survived'])
print(model.get_feature_importance(prettified=True))

# ________________ ПРЕДСКАЗАНИЕ ПО СОЗДАННОЙ МОДЕЛИ И СОХРАНЕНИЕ В ФАЙЛ ____________________ #
test['Survived'] = model.predict(test[['Pclass', 'Age', 'Sex', 'Name_1', 'Name_2',  'SibSp',  'Parch','Embarked',]])
test[['PassengerId', 'Survived']].to_csv('titanic_sub.csv', index=False)

print("Все")