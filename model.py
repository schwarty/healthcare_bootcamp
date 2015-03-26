import numpy as np

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import clone
from sklearn.grid_search import GridSearchCV


class MeetForester(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1., penalty='l2', percentile=60, n_estimators=100, n_jobs=1, random_state=None):
        self.percentile = percentile
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.base_estimator = Pipeline([
            ('imputer_most_frequent', Imputer(strategy='most_frequent')),
            ('select', SelectPercentile(f_classif, percentile)),
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)),
        ])

    def fit(self, X, y):
        self.estimator_ = clone(self.base_estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class MeetLogistic(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1., penalty='l2', percentile=60, random_state=None):
        self.C = C
        self.penalty = penalty
        self.percentile = percentile
        self.random_state = random_state

        self.base_estimator = Pipeline([
            ('imputer_most_frequent', Imputer(strategy='most_frequent')),
            ('scaler', StandardScaler()),
            ('select', SelectPercentile(f_classif, percentile)),
            ('clf', LogisticRegression(C=C, penalty=penalty, class_weight='auto', random_state=random_state)),
        ])

    def fit(self, X, y):
        self.estimator_ = clone(self.base_estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class AveragingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators):
        self.base_estimators = base_estimators

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimators_ = [clone(e).fit(X, y) for e in self.base_estimators]
        return self

    def predict_proba(self, X):
        return np.vstack([e.predict_proba(X)[np.newaxis] for e in self.estimators_]).mean(axis=0)

    def predict(self, X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1), axis=0)


Classifier = AveragingEnsemble

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('train.csv')
    df.head()
    print df.shape

    y = df['TARGET'].values
    X = df.drop('TARGET', axis=1).values

    random_state = 0

    rf = MeetForester(random_state=random_state)
    lr = MeetLogistic(random_state=random_state)

    clf = AveragingEnsemble([rf, lr])
    cv = StratifiedShuffleSplit(y, n_iter=5, random_state=random_state)

    # param_grid = {
    #     'C': np.logspace(-2, 2, 5),
    #     'penalty': ['l1', 'l2'],
    #     'n_estimators': [50, 100, 300],
    #     'percentile': [50, 60, 80],
    #     }

    # grid = GridSearchCV(clf, param_grid=param_grid, scoring='roc_auc', n_jobs=-1)
    # grid.fit(X, y)

    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print np.mean(scores)
