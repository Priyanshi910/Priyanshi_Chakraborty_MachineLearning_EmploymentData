# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 00:15:24 2020

@author: Priyanshi Chakrabort
"""

import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling

JS =pd.read_csv("job_skills.CSV")

JS.head()
JS.dtypes


print(JS.isnull().values.sum())

print(JS.isnull().sum())

JS = JS.fillna(JS['Responsibilities'].value_counts().index[0])
JS = JS.fillna(JS['Minimum Qualifications'].value_counts().index[0])
JS = JS.fillna(JS['Preferred Qualifications'].value_counts().index[0])

print(JS.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt
Category_count = JS['Category'].value_counts()
sns.set(style="darkgrid")
sns.barplot(Category_count.index, Category_count.values, alpha=0.9)
plt.title('Frequency Distribution of Job Categories')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()

from sklearn.preprocessing import LabelEncoder
JS_sklearn = JS.copy()
lb = LabelEncoder()
JS_sklearn['Title'] = lb.fit_transform(JS['Title'])

JS_sklearn.head()

JS_sklearn['Category'] = lb.fit_transform(JS['Category'])

JS_sklearn.head()

JS_sklearn['Location'] = lb.fit_transform(JS['Location'])

JS_sklearn.head()

JS_sklearn['Responsibilities'] = lb.fit_transform(JS['Responsibilities'])

JS_sklearn.head()

JS_sklearn['Minimum Qualifications'] = lb.fit_transform(JS['Minimum Qualifications'])

JS_sklearn.head()

JS_sklearn['Preferred Qualifications'] = lb.fit_transform(JS['Preferred Qualifications'])

JS_sklearn.head()

X1 = JS_sklearn.iloc[:, :-1].values 
Y1 = JS_sklearn.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.3, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)	

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)



print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

regressor = RandomForestRegressor(n_estimators=400, random_state=0)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)



print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

import matplotlib. pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
x, y = make_classification(n_samples=100,
                           n_features=6,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(x, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(JS_sklearn.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1,JS_sklearn.shape[1]])
plt.show()