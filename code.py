import pandas as pd
#logistic regression model
import math
%matplotlib inline 

from IPython.display import Image
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn

from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

plt.style.use('fivethirtyeight') # Good looking plots
pd.set_option('display.max_columns', None) # Display any number of columns

import seaborn as sns
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer 
import scipy as sp
import seaborn as sns
%matplotlib inline
nt = pd.read_excel('E:/finaldatamining.xlsx')
nt.head()
b = nt.fillna(nt.mean(),inplace = True)
c = b.drop(['Second HDL SI'],axis = 1)
d = c.drop(['Third HDL SI'],axis = 1)
e = d.drop(['Total Cholesterol( mg/dL)','Total Cholesterol( mmol/L)','PAT_ID'],axis = 1)
e.info()
#Steps Involved
'''
Steps Involved:
1. Pre- processing
2. Feature Engineering
3. Model Development
4. Testing
 '''
# Feature Engineering
'''Feature Selection: 
The dataset has 33 features in total. Feature selection will help us understand which features are useful 
for our model
nt.describe()
#Building Random Forest model
# Import libraries for building random forest regressor
from sklearn.ensemble import RandomForestRegressor
# Import metrics for accuracy score calculation
from sklearn.metrics import roc_auc_score
# Defining the outcome variable according to the risk levels. Total cholesterol>200 = High risk and vice versa
outcome = nt['Total Cholesterol( mg/dL)']>200
outcome = outcome.astype(int)
bins = [69, 200, 240, 813]
group_names = ['0', '1', '2']
categories = pd.cut(nt['Total Cholesterol( mg/dL)'], bins, labels=group_names)
nt['categories'] = pd.cut(nt['Total Cholesterol( mg/dL)'], bins, labels=group_names)
x=categories.astype(int)
y=x.fillna(x.mean())
y=pd.DataFrame(y)
for i, col in enumerate(y.columns.tolist(), 1):
    y.loc[:, col] *= i
y = y.sum(axis=1)
corr = ntn.corr()
sns.heatmap(corr,vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
            import sklearn.preprocessing as sp
y1 = sp.label_binarize(y1, classes=[0, 1, 2])
model = RandomForestRegressor(n_estimators = 500, oob_score = True, random_state = 42)
model.fit(ndnd,y1)
model.oob_score_
m = model.oob_prediction_
roc_auc_score(y1,m)
feature_importances = pd.Series(model.feature_importances_, index = e.columns)
sns.set()
sns.set_style('whitegrid')
feature_importances.sort_values().plot(kind = 'bar',figsize = (10,6))
plt.plot(np.sort(model.feature_importances_))
def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred
    print('Passive Aggressive Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))))
print('Gradient Boosting Classifier:  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support vector machine(SVM):   {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:      {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:           {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))))
print('Dump Classifier: {:.2f}'.format(metrics.accuracy_score(y, [0 for ii in y.tolist()])))
pass_agg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))
grad_ens_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))
decision_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, tree.DecisionTreeClassifier))
ridge_clf_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.RidgeClassifier))
svm_svc_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, svm.SVC))
random_forest_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.RandomForestClassifier))
k_neighbors_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))
logistic_reg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.LogisticRegression))
dumb_conf_matrix = metrics.confusion_matrix(y, [0 for ii in y.tolist()]); # ignore the warning as they are all 0

conf_matrix = {
                1: {
                    'matrix': pass_agg_conf_matrix,
                    'title': 'Passive Aggressive',
                   },
                2: {
                    'matrix': grad_ens_conf_matrix,
                    'title': 'Gradient Boosting',
                   },
                3: {
                    'matrix': decision_conf_matrix,
                    'title': 'Decision Tree',
                   },
                4: {
                    'matrix': ridge_clf_conf_matrix,
                    'title': 'Ridge',
                   },
                5: {
                    'matrix': svm_svc_conf_matrix,
                    'title': 'Support Vector Machine',
                   },
                6: {
                    'matrix': random_forest_conf_matrix,
                    'title': 'Random Forest',
                   },
                7: {
                    'matrix': k_neighbors_conf_matrix,
                    'title': 'K Nearest Neighbors',
                   },
                8: {
                    'matrix': logistic_reg_conf_matrix,
                    'title': 'Logistic Regression',
                   },
                9: {
                    'matrix': dumb_conf_matrix,
                    'title': 'Dumb',
                   },
}
fix, ax = plt.subplots(figsize=(16, 12))
plt.suptitle('Confusion Matrix of Various Classifiers')
for ii, values in conf_matrix.items():
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 3, ii) # starts from 1
    plt.title(title);
    sns.heatmap(matrix, annot=True,  fmt='');
    print('Passive Aggressive Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))))
print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support vector machine(SVM):\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, linear_model.LogisticRegression))))
print('Dump Classifier:\n {}\n'.format(metrics.classification_report(y, [0 for ii in y.tolist()]))); # ignore the warning as they are all 0
gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X, y)
feature_importances = pd.Series(gbc.feature_importances_, index = ntn7.columns)
feature_importances.sort_values()
bins = [68, 200, 240, 815]
group_names = ['0', '1', '2']
categories = pd.cut(ndn['Total Cholesterol( mg/dL)'], bins, labels=group_names)
ndn['categories'] = pd.cut(ndn['Total Cholesterol( mg/dL)'], bins, labels=group_names)
x1=categories.astype(int)
y1=x1.fillna(x1.mean())
y1=pd.DataFrame(y1)
for i, col in enumerate(y1.columns.tolist(), 1):
    y1.loc[:, col] *= i
y1 = y1.sum(axis=1)
parameters = {
              'n_estimators': 500, 
              'max_depth': 3,
              'learning_rate': 0.02, 
              'loss': 'ls'
             }
from sklearn import ensemble
from sklearn import metrics
classifier = ensemble.GradientBoostingRegressor(**parameters)

classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)
mse = metrics.mean_squared_error(Y_test, predictions)
print('Mean Square Error: {:.3f}'.format(mse))
from sklearn.metrics import label_ranking_average_precision_score
label_ranking_average_precision_score(Y_test, predictions) 
plt.figure(figsize=(16, 12))

plt.scatter(range(predictions.shape[0]), predictions, label='predictions', c='#348ABD', alpha=0.4)
plt.scatter(range(Y_test.shape[0]), Y_test, label='actual values', c='#A60628', alpha=0.4)
plt.ylim([Y_test.min(), predictions.max()])
plt.xlim([0, predictions.shape[0]])
plt.legend();
test_score = [classifier.loss_(Y_test, Y_pred) for Y_pred in classifier.staged_decision_function(X_test)]

plt.figure(figsize=(16, 12))
plt.title('Deviance');
plt.plot(np.arange(parameters['n_estimators']) + 1, classifier.train_score_, c='#348ABD',
         label='Training Set Deviance');
plt.plot(np.arange(parameters['n_estimators']) + 1, test_score, c='#A60628',
         label='Test Set Deviance');
plt.annotate('Overfit Point', xy=(820, test_score[820]), xycoords='data',
            xytext=(420, 0.06), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )
plt.legend(loc='upper right');
plt.xlabel('Boosting Iterations');
plt.ylabel('Deviance');
# Get Feature Importance from the classifier
feature_importance = classifier.feature_importances_
# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(16, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
plt.yticks(pos, np.asanyarray(datafin.columns)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
