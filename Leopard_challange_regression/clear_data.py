import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier
from datetime import datetime
from phik import phik_matrix


sample_sub = pd.read_csv('sample_submission.csv')
test_real = pd.read_csv('test.csv')
train_real = pd.read_csv('train.csv')

def clear_date(x):
    day, month, year = [int(i) for i in x['Date'].split("/")]
    start = datetime(year, month, day)
    x['Date'] = (datetime(2020,1,1) - start).days
    return x

# print(train_real.isna().mean())     # чтобы посмотреть, где сколько нулей
test_real = test_real.apply(clear_date, axis=1)
train_real['Address'] = train_real['Address'].apply(lambda x: x.split()[1])

cols_all = ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount', 'id']

create_bathroom = True
create_bedroom2 = True
create_car = True

if create_bathroom:
    """Добавление недостающий значений bathroom"""
    has_bathroom = test_real[test_real['Bathroom'].notnull()].copy()
    not_bathroom = test_real[test_real['Bathroom'].isna()].copy()
    train_x, test_x = train_test_split(has_bathroom, train_size=0.8, random_state=42)

    #phik_ower = train_bathroom[['Rooms', 'Type', 'Price', 'Method', 'SellerG','Date', 'Distance', 'Postcode', 'Bathroom']].phik_matrix()
    #print(phik_ower['Bathroom'].sort_values(ascending=False))

    X = ["Rooms", 'Type', 'SellerG', 'Method',  "Date", "Distance", "Postcode"]
    Y = ["Bathroom" ]
    cat = ["Rooms", 'Type', 'Method', 'SellerG']
    params = {'verbose': 100,
              "random_seed": 42,
              'learning_rate': 0.05}

    classifyer = False

    if classifyer:
        bath_model = CatBoostClassifier(**params, cat_features=cat, )
        bath_model.fit(train_x[X], train_x[Y], eval_set=(test_x[X], test_x[Y]))
        #a = bath_model.predict_proba(test_bathroom[X])
        #print("all")

        def get_max_num(x):
            c = pd.concat([x.transpose(), pd.DataFrame(range(0, len(x)))], axis=1)
            c.columns = ['pred', 'num']
            return  int(c[c['pred'] == c['pred'].max()]['num'])

        def get_and_paste_predict(train_df_x, pred_x):
            pred_x.index = train_df_x.index
            return pred_x.apply(get_max_num, axis=1)

        pred_provb = pd.DataFrame(bath_model.predict_proba(not_bathroom[X]))
        not_bathroom[f'Bathroom'] =get_and_paste_predict(not_bathroom, pred_provb)
        train_real = pd.concat([has_bathroom, not_bathroom])

    else:
        bath_model = CatBoostRegressor(**params, cat_features=cat)
        bath_model.fit(train_x[X], train_x[Y], eval_set=(test_x[X], test_x[Y]))
        pred_x = pd.DataFrame(bath_model.predict(not_bathroom[X]))
        pred_x.index = not_bathroom.index
        not_bathroom['Bathroom'] = pred_x
        test_real = pd.concat([has_bathroom, not_bathroom])


if create_bedroom2:
    """Добавление недостающий значений bathroom"""
    has_bedroom2 = test_real[test_real['Bedroom2'].notnull()].copy()
    not_bedroom2 = test_real[test_real['Bedroom2'].isna()].copy()
    train_x, test_x = train_test_split(has_bedroom2, train_size=0.6, random_state=42)

    # phik_ower = train_x[['Rooms', 'Type','Method', 'SellerG','Date', 'Distance', 'Postcode', 'Bedroom2']].phik_matrix()
    # print(phik_ower['Bedroom2'].sort_values(ascending=False))

    X = ["Rooms", 'Type', 'SellerG', 'Method', "Date", "Distance", "Postcode"]
    Y = ["Bedroom2"]
    cat = ["Rooms", 'Type', 'Method', 'SellerG']
    params = {'verbose': 100,
              "random_seed": 42,
              'learning_rate': 0.05}

    bath_model = CatBoostRegressor(**params, cat_features=cat, )
    bath_model.fit(train_x[X], train_x[Y], eval_set=(test_x[X], test_x[Y]))

    pred_x = pd.DataFrame(bath_model.predict(not_bedroom2[X]))
    pred_x.index = not_bedroom2.index
    not_bedroom2['Bedroom2'] = pred_x
    test_real = pd.concat([has_bedroom2, not_bedroom2])


if create_car:
    """Добавление недостающий значений bathroom"""
    has_car = test_real[test_real['Car'].notnull()].copy()
    not_car= test_real[test_real['Car'].isna()].copy()
    train_x, test_x = train_test_split(has_car, train_size=0.6, random_state=42)

    # phik_ower = train_x[['Rooms', 'Type','Method', 'SellerG','Date', 'Distance', 'Postcode', 'Car']].phik_matrix()
    # print(phik_ower['Car'].sort_values(ascending=False))

    X = ["Rooms", 'Type', 'SellerG', 'Method', "Date", "Distance", "Postcode"]
    Y = ["Car"]
    cat = ["Rooms", 'Type', 'Method', 'SellerG']
    params = {'verbose': 100,
              "random_seed": 42,
              'learning_rate': 0.05}

    bath_model = CatBoostRegressor(**params, cat_features=cat, )
    bath_model.fit(train_x[X], train_x[Y], eval_set=(test_x[X], test_x[Y]))

    pred_x = pd.DataFrame(bath_model.predict(not_car[X]))
    pred_x.index = not_car.index
    not_car['Car'] = pred_x
    test_real = pd.concat([has_car, not_car])
test_real.to_csv('test_add_bath_bed_car.csv', index=False)
print('Все')