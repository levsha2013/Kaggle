from sklearn.model_selection import train_test_split
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


random_state = 42
iters = 1000

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_orig = pd.read_csv('trainn.csv')
test_orig = pd.read_csv('testt.csv')

# выделяем признаки для обучения и terget
X = test.columns.drop('id')
Y = 'prognosis'

# в оригинале и то и другое с target. Можно все это объеденить для генерации итога
#  только target там с пробелом - сейчас исправим
train_orig[Y] = train_orig[Y].apply(lambda x: x.replace(" ", "_"))
test_orig[Y] = test_orig[Y].apply(lambda x: x.replace(" ", "_"))
train_full = pd.concat([train, train_orig, test_orig])


# PCA практически не уменьшает количество признаков: 99% =-4 признака, 99.99% вообще ничего не убирает
# from sklearn.decomposition import PCA
# + какая-то ошибка. Но все равно признаки не убираются - не используем PCA
"""pca = PCA(0.9999)
pca.fit(pd.concat([train_full[X], test[X]]))
train_full = pd.concat([pd.DataFrame(pca.transform(train_full[X])), train_full[Y]], axis=1)
test = pd.concat([test['id'], pd.DataFrame(pca.transform(test[X]))], axis=1)
X = train.columns.drop(Y)"""

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
for target_x in train_full[Y].unique():
    train_target = train_full[train_full[Y] == target_x]
    n_components = range(1,15)
    models = [GaussianMixture(n, covariance_type='full').fit(train_target[X]) for n in n_components]
    plt.plot(n_components, [m.bic(train_target[X]) for m in models], label = 'BIC', color = 'blue', alpha=0.5)
    plt.plot(n_components, [m.aic(train_target[X]) for m in models], label = 'AIC', color = 'red', alpha=0.5)
    #plt.show()
    #choice_n_comp = int(input(f"Введи уже количество компонент, которое выбрал для параметра {target_x}!"))
    choice_n_comp = 1

    gmm = GaussianMixture(n_components=choice_n_comp, covariance_type='full', random_state=random_state)
    gmm.fit(train_target[X])
    train_new = pd.DataFrame(gmm.sample(500)[0], columns=X)
    train_full = pd.concat([train_full, train_new])

train_1, test_1 = train_test_split(train_full, test_size=0.2, random_state=random_state, shuffle=True)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
grid = GridSearchCV(RandomForestClassifier(random_state=random_state),
                    {"n_estimators": [i for i in range(20,500,20)]}, n_jobs=-1, cv=3)
grid.fit(train_full[X], train_full[Y])
print(grid.best_params_)


print("Все")