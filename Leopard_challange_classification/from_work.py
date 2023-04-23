import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from tqdm import tqdm
random_state = 42
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['oral'] = train['oral'].apply(lambda x: 1 if 'Y' else 'N')
train['tartar'] = train['tartar'].apply(lambda x: 1 if 'Y' else 'N')

test['oral'] = test['oral'].apply(lambda x: 1 if 'Y' else 'N')
test['tartar'] = test['tartar'].apply(lambda x: 1 if 'Y' else 'N')

Y = 'smoking'
X = train.columns.drop(['ID', 'smoking'])

from sklearn.decomposition import PCA
pca = PCA(0.9999)
pca.fit(pd.concat([train[X], test[X]]))
train = pd.concat([pd.DataFrame(pca.transform(train[X])), train[Y]], axis=1)
test = pd.concat([test['ID'], pd.DataFrame(pca.transform(test[X]))], axis=1)
X = train.columns.drop('smoking')

from sklearn.mixture import GaussianMixture

cur = train[train[Y]==1]
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=random_state)
gmm.fit(cur[X])
cur_new = pd.DataFrame(gmm.sample(11063)[0], columns=X)
cur_new[Y] = 1
train_XX = train
train = pd.concat([train, cur_new])


train_1, test_1 = train_test_split(train, test_size=0.2, random_state=random_state, shuffle=True)

"""import matplotlib.pyplot as plt
for i in X:
    for j in X:
        if i == j: continue
        plt.scatter(train[i], train[j], alpha=0.5, c=train[Y], cmap=plt.cm.get_cmap('cubehelix', 2))
        plt.colorbar(ticks=range(30), label=f"{i}_{j}")
        plt.show()"""


#model = GridSearchCV(LogisticRegression(random_state=random_state, max_iter=10000, class_weight=None, dual=True,
#                                        fit_intercept=True,multi_class='auto', solver='liblinear'),
#                     {'C': [0.01,0.1,1,10,100,1000,3000], 'intercept_scaling':[0.1,1,5,10,20,50,100]}, n_jobs=4, cv=3, verbose=2) # {'C': 1, 'intercept_scaling': 100}
#model.fit(train[X], train[Y])
#print(model.best_params_)

model = LogisticRegression(max_iter=10000, random_state=random_state)
model.fit(train_1[X], train_1[Y])
pred = model.predict(train_XX[X])
model2 = LogisticRegression(random_state=random_state, max_iter=10000, class_weight=None, dual=True, C=1,
                            fit_intercept=True,multi_class='auto', solver='liblinear', intercept_scaling=100,)
model2.fit(train_1[X], train_1[Y])
pred2 = model2.predict(test_1[X])
print(roc_auc_score(train_XX[Y], pred))
print(roc_auc_score(test_1[Y], pred2))
print()
"""model2 = LogisticRegression(random_state=random_state, max_iter=10000, class_weight=None, dual=True, C=1,
                            fit_intercept=True,multi_class='auto', solver='liblinear', intercept_scaling=100)
model2.fit(train[X], train[Y])
test[Y] = model2.predict(test[X])
test[['ID', 'smoking']].to_csv('LogisticRegression_best.csv', index=False)"""




model = SVC(random_state=random_state, C=1)
model.fit(train_1[X], train_1[Y])
pred = model.predict(test_1[X])
model2 = SVC(random_state=random_state, kernel='rbf', )
model2.fit(train_1[X], train_1[Y])
pred2 = model2.predict(test_1[X])
print(roc_auc_score(test_1[Y], pred))
print(roc_auc_score(test_1[Y], pred2))
print()
"""model2 = SVC(random_state=random_state, kernel='rbf', class_weight='balanced', C=0.001)
model2.fit(train[X], train[Y])
test['smoking'] = model2.predict(test[X])
test[['ID', 'smoking']].to_csv('SVC_best.csv', index=False)"""



#model = GridSearchCV(RandomForestClassifier(random_state=random_state),
#                     {'n_estimators': [36,38,40], 'max_leaf_nodes': [2008,2012,2015], 'max_depth': [28, 30,32]
#                      }, n_jobs=4, cv=3, verbose=3)
#model.fit(train[X], train[Y])
#print(model.best_params_)
# {'max_depth': 15, 'max_leaf_nodes': 20, 'min_samples_split': 2, 'n_estimators': 50}
# {'max_depth': 20, 'max_leaf_nodes': 30, 'min_samples_split': 2, 'n_estimators': 50}
# {'max_depth': 20, 'max_leaf_nodes': 35, 'n_estimators': 40}
# {'max_depth': 19, 'max_leaf_nodes': 40, 'n_estimators': 40}
# {'max_leaf_nodes': 200, 'n_estimators': 42}
# {'max_depth': 30, 'max_leaf_nodes': 2012, 'n_estimators': 38} last best
model = RandomForestClassifier(random_state=random_state, n_jobs=4)
model.fit(train_1[X], train_1[Y])
pred = model.predict(train_XX[X])
model2 = RandomForestClassifier(random_state=random_state, max_depth=30, min_samples_split=2, max_leaf_nodes=2012,
                                 n_estimators=38)
model2.fit(train_1[X], train_1[Y])
pred2 = model2.predict(test_1[X])
print(roc_auc_score(train_XX[Y], pred))
print(roc_auc_score(test_1[Y], pred2))
print()

"""model2 = RandomForestClassifier(random_state=random_state, max_depth=30, min_samples_split=2, max_leaf_nodes=2012,
                                 n_estimators=38)
model2.fit(train[X], train[Y])
test['smoking'] = model2.predict(test[X])
test[['ID', 'smoking']].to_csv('Rand_Forest_best.csv', index=False)"""


#model = GridSearchCV(AdaBoostClassifier(random_state=random_state),
#                     {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [119,120,121]}, n_jobs=4, cv=3, verbose=2)
#model.fit(train[X], train[Y])
#print(model.best_params_)
# {'algorithm': 'SAMME.R', 'n_estimators': 100}
# {'algorithm': 'SAMME.R', 'n_estimators': 125}
# {'algorithm': 'SAMME.R', 'n_estimators': 120} top
model = AdaBoostClassifier(random_state=random_state)
model.fit(train_1[X], train_1[Y])
pred = model.predict(train_XX[X])
model2 = AdaBoostClassifier(random_state=random_state, n_estimators=125, algorithm='SAMME.R')
model2.fit(train_1[X], train_1[Y])
pred2 = model2.predict(test_1[X])
print(roc_auc_score(train_XX[Y], pred))
print(roc_auc_score(test_1[Y], pred2))
print()

"""model2 = AdaBoostClassifier(random_state=random_state, n_estimators=125, algorithm='SAMME.R')
model2.fit(train[X], train[Y])
test['smoking'] = model2.predict(test[X])
test[['ID', 'smoking']].to_csv('Ada_sub_best.csv', index=False)"""


print("Все")
