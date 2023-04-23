import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, cv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ________________ ЧТЕНИЕ CSV ____________________ #
sample_sub = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
random_state = 50

# ________________ ПЕРВИЧНАЯ ОБРАБОТКА DATA и ADDRESS ____________________ #
# функция переводит дату в количество дней до 2022 01 01


def clear_data(x):
    day, month, year = [int(i) for i in x['Date'].split("/")]
    start = datetime(year, month, day)
    x['Date'] = (datetime(2020,1,1) - start).days
    return x


train['Address'] = train['Address'].apply(lambda x: x.split()[1])       # преобразование адреса - убирание домов
train = train.apply(clear_data, axis=1)                               # преобразование даты
test['Address'] = test['Address'].apply(lambda x: x.split()[1])     # преобразование адреса - убирание домов
test = test.apply(clear_data, axis=1)                               # преобразование даты


# ________________ Изменение конкретных категориальных фичей ____________________ #
train.loc[train['Suburb'] == 'Footscray', 'Propertycount'] = 7570.0                   # закончил Propertycount
train.loc[train['Postcode'] == 3011.0, 'CouncilArea'] = 'Maribyrnong City Council'    # закончил CouncilArea
train.loc[train['Postcode'] == 3011.0, 'Regionname'] = 'Western Metropolitan'         # закончил Regionname


test.loc[test['Postcode'] == 3124, 'Propertycount'] = 8920.0
test.loc[test['Suburb'] == 'Fawkner Lot', 'Postcode'] =3060.0
test.loc[test['Suburb'] == 'Fawkner Lot', 'Propertycount'] = 3060.0
test.loc[test['Suburb'] == 'Fawkner Lot', 'CouncilArea'] = 'Hume City Council'
test.loc[test['Suburb'] == 'Fawkner Lot', 'Regionname'] = 'Northern Metropolitan'
test.loc[test['Suburb'] == 'Camberwell', 'CouncilArea'] = 'Boroondara City Council'
test.loc[test['Suburb'] == 'Camberwell', 'Regionname'] = 'Southern Metropolitan'

# ________________ Ограничение численных фичей ____________________ #
train = train.drop(train[train['YearBuilt'] < 1500].index)              # выброс по году
train = train.drop(train[train['BuildingArea'] > 10000].index)          # 10 k выбросы  (max = 2k - )
train = train.drop(train[train['Landsize'] > 50000].index)              # 50k выброы (max = 15k - 17)
train = train.drop(train[train['Bedroom2'] > 10].index)              # выброы (max = 15k - 17)


def paste_Latt(x):
    x.loc[x['Lattitude'].isna(), 'Lattitude'] = x['Lattitude'].mean()
    return x


def paste_Long(x):
    x.loc[x['Longtitude'].isna(), 'Longtitude'] = x['Longtitude'].mean()
    return x


train = train.groupby('CouncilArea').apply(paste_Latt)
train = train.groupby('CouncilArea').apply(paste_Long)
test = test.groupby('CouncilArea').apply(paste_Latt)
test = test.groupby('CouncilArea').apply(paste_Long)

c = train[['Lattitude', 'Longtitude', 'Propertycount', 'Regionname', 'CouncilArea']]
d = train[['Lattitude', 'Longtitude', 'Propertycount']]


#mms = MinMaxScaler()
#mms.fit(d)
#d = pd.DataFrame(mms.transform(d))
print("Все")