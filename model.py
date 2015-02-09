from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import clone

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



def model(X_train, y_train, X_test):
    clf = model_spec()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score


def model_spec():
    class BlendingEnsemble(BaseEstimator, ClassifierMixin):

        def __init__(self, estimators, blender=None):
            self.estimators = estimators
            self.blender = blender

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.estimators_ = [clone(e).fit(X, y) for e in self.estimators]
            new_features = np.hstack([e.predict_proba(X)[:, 1:] for e in self.estimators_])
            blender = LogisticRegression(C=1) if self.blender is None else self.blender
            self.blender_ = clone(blender).fit(new_features, y)
            return self

        def predict_proba(self, X):
            new_features = np.hstack([e.predict_proba(X)[:, 1:] for e in self.estimators_])
            return self.blender_.predict_proba(new_features)

        def predict(self, X):
            proba = self.predict_proba(X)
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)


        clf1 = Pipeline([('imputer', Imputer(strategy='most_frequent')),
                         ('scaler', StandardScaler()),
                         ('select', SelectPercentile(f_classif, 90)),
                         ('clf', AdaBoostClassifier(RandomForestClassifier(n_estimators=300, max_depth=3, n_jobs=-1), n_estimators=20)),
])

        clf2 = Pipeline([('imputer', Imputer(strategy='most_frequent')),
                         ('scaler', StandardScaler()),
                         ('select', SelectPercentile(f_classif, 50)),
                         ('clf', LogisticRegression(C=.01, penalty='l2')),
])

        clf = BlendingEnsemble([clf1, clf2])
        return clf
