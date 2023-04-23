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


# ________________ Изменение конкретных категориальных фичей ____________________ #
train_real.loc[train_real['Suburb'] == 'Footscray', 'Propertycount'] = 7570.0                   # закончил Propertycount
train_real.loc[train_real['Postcode'] == 3011.0, 'CouncilArea'] = 'Maribyrnong City Council'    # закончил CouncilArea
train_real.loc[train_real['Postcode'] == 3011.0, 'Regionname'] = 'Western Metropolitan'         # закончил Regionname


test_real.loc[test_real['Postcode'] == 3124, 'Propertycount'] = 8920.0
test_real.loc[test_real['Suburb'] == 'Fawkner Lot', 'Postcode'] =3060.0
test_real.loc[test_real['Suburb'] == 'Fawkner Lot', 'Propertycount'] = 3060.0
test_real.loc[test_real['Suburb'] == 'Fawkner Lot', 'CouncilArea'] = 'Hume City Council'
test_real.loc[test_real['Suburb'] == 'Fawkner Lot', 'Regionname'] = 'Northern Metropolitan'
test_real.loc[test_real['Suburb'] == 'Camberwell', 'CouncilArea'] = 'Boroondara City Council'
test_real.loc[test_real['Suburb'] == 'Camberwell', 'Regionname'] = 'Southern Metropolitan'

# ________________ Ограничение численных фичей ____________________ #
train_real = train_real.drop(train_real[train_real['YearBuilt'] < 1500].index)              # выброс по году
train_real = train_real.drop(train_real[train_real['BuildingArea'] > 2000].index)          # 10 k выбросы  (max = 2k - )
train_real = train_real.drop(train_real[train_real['Landsize'] > 15000].index)              # 50k выброы (max = 15k - 17)
train_real = train_real.drop(train_real[train_real['Bedroom2'] > 10].index)              # выброы (max = 15k - 17)

def paste_Latt(x):
    x.loc[x['Lattitude'].isna(), 'Lattitude'] = x['Lattitude'].mean()
    return x


def paste_Long(x):
    x.loc[x['Longtitude'].isna(), 'Longtitude'] = x['Longtitude'].mean()
    return x


train_real = train_real.groupby('CouncilArea').apply(paste_Latt)
train_real = train_real.groupby('CouncilArea').apply(paste_Long)
test_real = test_real.groupby('CouncilArea').apply(paste_Latt)
test_real = test_real.groupby('CouncilArea').apply(paste_Long)


"""# ________________ ЛОГАРИФМИЗАЦИЯ ЧИСЛЕННЫХ ПРИЗНАКОВ ____________________ #
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
test_real[numbers_feaches] = pd.DataFrame(scaler.transform(test_real[numbers_feaches]), columns=numbers_feaches)"""
def predict_BuildingArea_nan(df):
    # предсказание на основе Rooms, Distance и Landsize
    y = ['BuildingArea']
    x = ['Rooms', 'Distance', 'Landsize']
    train_to_build_area = df[['Rooms', 'Distance', 'Landsize', 'BuildingArea']].dropna()
    #train_to_build_area = train_to_build_area.drop(train_to_build_area[train_to_build_area['BuildingArea']==0].index)
    train, test = train_test_split(train_to_build_area, random_state=random_state, train_size=0.8)
    get_building = CatBoostRegressor(random_seed=random_state, eval_metric='MAPE', loss_function='RMSE',
                                     learning_rate=0.003,
                                     iterations=5000, verbose=1000)
    get_building.fit(train[x], train[y], eval_set=(test[x], test[y]))
    n = get_building.best_iteration_ + 1

    get_building = CatBoostRegressor(random_seed=random_state, eval_metric='MAPE', loss_function='RMSE', iterations=n,
                                     learning_rate=0.003, verbose=1000)
    get_building.fit(train_to_build_area[x], train_to_build_area[y])
    df.loc[df['BuildingArea'].isna(), 'BuildingArea'] = get_building.predict( df[df['BuildingArea'].isna()][x])
    #df.loc[df['BuildingArea'] == 0, 'BuildingArea'] = get_building.predict(df[df['BuildingArea'] == 0][x])

    return df

#train_real = predict_BuildingArea_nan(train_real)
#test_real = predict_BuildingArea_nan(test_real)


# ________________ ВЫДЕЛЕНИЕ ПРИЗНАКОВ, НАСТРОЙКА И ОБУЧЕНИЕ ПЕРВИЧНОЙ МОДЕЛИ ____________________ #
Y = ['Price']
X = ["Suburb", 'Address', 'Rooms', 'Type', 'Method', 'SellerG', 'Distance', 'Date', 'Postcode',
     'Bathroom', 'Bedroom2', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount', 'Regionname', 'CouncilArea']

cat_featches = [ "Suburb", 'Address', 'Method', 'SellerG', 'Type', 'Regionname', 'CouncilArea']

train, test = train_test_split(train_real, random_state=random_state, train_size=0.8)
"""val, test = train_test_split(test, random_state=random_state, train_size=0.5 )


train_data = Pool(data=train[X],
                  label=train[Y],
                  cat_features=cat_featches)

valid_data = Pool(data=val[X],
                  label=val[Y],
                  cat_features=cat_featches)

test_data = Pool(data=test[X],
                 label=test[Y],
                 cat_features=cat_featches)

params = {'verbose': 100,
          'eval_metric': 'MAPE',
          'loss_function': 'RMSE',
          'random_seed': random_state,
          'learning_rate': 0.15
          }

model = CatBoostRegressor(**params)
model.fit(train_data, eval_set=valid_data)
test['price_pred'] = model.predict(test[X])


# ________________ ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ ____________________ #

best_iter = model.best_iteration_ + 1
params['iterations'] = best_iter

model = CatBoostRegressor(**params)
train_full = pd.concat([train, val])
train_full_data = Pool(data=train_full[X],
                       label=train_full[Y],
                       cat_features=cat_featches)
# model.fit(train_full_data, eval_set=test_data)"""


# ________________ КРОСС-ВАЛИДАЦИЯ - нахождение лучшей итерации____________________ #
params = {'iterations': 4000,
          'verbose': 100,
          'eval_metric': 'MAPE',
          'loss_function': 'RMSE',
          'random_seed': random_state,
          'learning_rate': 0.05
          }

train_data = Pool(data=train[X],
                  label=train[Y],
                  cat_features=cat_featches)
test_data = Pool(data=test[X],
                 label=test[Y],
                 cat_features=cat_featches)

cv_data = cv(params=params, pool=train_data, fold_count=5, shuffle=True,
             partition_random_seed=random_state, stratified=False, verbose=False)
name_of_metrix = f"test-{params['eval_metric']}-mean"
best_iter = cv_data[cv_data[name_of_metrix] == cv_data[name_of_metrix].min()]['iterations'].values[0]

# ________________ КРОСС-ВАЛИДАЦИЯ - обучение до лучшей итерации____________________ #
params['iterations'] = best_iter
model = CatBoostRegressor(**params)
model.fit(train_data)
print(model.get_feature_importance(prettified=True))

# ________________ ПРЕДСКАЗАНИЕ ПО СОЗДАННОЙ МОДЕЛИ И СОХРАНЕНИЕ В ФАЙЛ ____________________ #
test_real['Price'] = model.predict(test_real[X])
test_real[['id', 'Price']].to_csv('my_sub.csv', index=False)