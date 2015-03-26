import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn import clone

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit



def model(X_train, y_train, X_test):
    clf = model_spec()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score
        

def model_spec():
        return Pipeline([('imputer', Imputer(strategy='most_frequent')),
                         ('scaler', StandardScaler()),
                         ('select', SelectPercentile(f_classif, 85)),
                         ('clf', BaggingClassifier(LogisticRegression(C=.01, penalty='l2'), n_estimators=100,
                                                   bootstrap_features=True, n_jobs=-1)),
                      ])


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('train.csv')
    df.head()
    print df.shape

    y = df['TARGET'].values
    X = df.drop('TARGET', axis=1).values

    clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
                         ('scaler', StandardScaler()),
                         ('select1', SelectPercentile(f_classif, 60)),
                         ('select2', RFECV(LogisticRegression(C=.01, penalty='l2', class_weight='auto'), scoring='roc_auc')),
                         ('clf', LogisticRegression(C=.01, penalty='l2', class_weight='auto')),
                      ])

    cv = StratifiedShuffleSplit(y, n_iter=5, random_state=0)

    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print np.mean(scores)
