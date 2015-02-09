from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline



def model(X_train, y_train, X_test):
    clf = model_spec()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score


def model_spec():
    return Pipeline([('imputer', Imputer(strategy='most_frequent')),
                    ('scaler', StandardScaler()),
                    ('select', SelectPercentile(f_classif, 90)),
                    ('clf', AdaBoostClassifier(RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1), n_estimators=20))
                 ])
