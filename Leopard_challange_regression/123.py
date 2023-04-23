import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, cv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ________________ ЧТЕНИЕ CSV ____________________ #
sample_sub = pd.read_csv('sample_submission.csv')
test_real = pd.read_csv('test.csv')
train_real = pd.read_csv('train.csv')
random_state = 50

# ________________ ПЕРВИЧНАЯ ОБРАБОТКА DATA и ADDRESS ____________________ #
# функция переводит дату в количество дней до 2022 01 01
def clear_data(x):
    day, month, year = [int(i) for i in x['Date'].split("/")]
    start = datetime(year, month, day)
    x['Date'] = (datetime(2020,1,1) - start).days
    return x
train_real['Address'] = train_real['Address'].apply(lambda x: x.split()[1])     # преобразование адреса - убирание домов
train_real = train_real.apply(clear_data, axis=1)                               # преобразование даты
test_real['Address'] = test_real['Address'].apply(lambda x: x.split()[1])     # преобразование адреса - убирание домов
test_real = test_real.apply(clear_data, axis=1)                               # преобразование даты


# ________________ ЛОГАРИФМИЗАЦИЯ ЧИСЛЕННЫХ ПРИЗНАКОВ ____________________ #
from math import log
numbers_feaches = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea',
                    'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount', 'Date']
def log_numeric(x):
    if x != 0:
        return x/abs(x)*log(abs(x))
    else:
        return 0

for i in numbers_feaches:
    test_real[i] = test_real[i].apply(log_numeric)
    train_real[i] = train_real[i].apply(log_numeric)


# ________________ НОРМАЛИЗАЦИЯ ПРИЗНАКОВ ____________________ #

mms = MinMaxScaler()
numbers_feaches =  ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea',
                    'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount', 'Date']


mms.fit(train_real[numbers_feaches])
train_real[numbers_feaches] = pd.DataFrame(mms.transform(train_real[numbers_feaches]), columns=numbers_feaches)
test_real[numbers_feaches] = pd.DataFrame(mms.transform(test_real[numbers_feaches]), columns=numbers_feaches)


# ________________ СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ ____________________ #

scaler = StandardScaler()
numbers_feaches = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea',
                    'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount', 'Date']


scaler.fit(train_real[numbers_feaches])
train_real[numbers_feaches] = pd.DataFrame(scaler.transform(train_real[numbers_feaches]), columns=numbers_feaches)
test_real[numbers_feaches] = pd.DataFrame(scaler.transform(test_real[numbers_feaches]), columns=numbers_feaches)

# ________________ Изменение конкретных категориальных фичей ____________________ #

train_real.loc[train_real['Postcode'] == 3011.0, 'CouncilArea'] = 'Maribyrnong City Council'
train_real.loc[train_real['Postcode'] == 3011.0, 'Regionname'] = 'Western Metropolitan'
train_real.loc[train_real['Suburb'] == 'Footscray', 'Propertycount'] = 7570.0
train_real.loc[train_real['CouncilArea'].isna(), 'CouncilArea'] = 'Maribyrnong City Council'
train_real.loc[train_real['Regionname'].isna(), 'Regionname'] = 'Western Metropolitan'


test_real.loc[test_real['Postcode'] == 3124.0, 'CouncilArea'] = 'Boroondara City Council'
test_real.loc[test_real['Postcode'] == 3124.0, 'Regionname'] = 'Southern Metropolitan'
test_real.loc[test_real['Postcode'] == 3124.0, 'Propertycount'] = 8920.0
test_real.loc[test_real['Suburb'] == 'Fawkner Lot', 'CouncilArea'] = 'Whitehorse City Council'
test_real.loc[test_real['Suburb'] == 'Fawkner Lot', 'Regionname'] = 'Eastern Metropolitan'
test_real.loc[test_real['Regionname'].isna(), 'CouncilArea'] = 'Boroondara City Council'
test_real.loc[test_real['Regionname'].isna(), 'Regionname'] = 'Southern Metropolitan'

print("Rjytw")