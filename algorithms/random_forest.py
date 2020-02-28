import skopt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def get_classifier():
    return Pipeline([
        ('anova', SelectKBest(f_classif, k=14)),
        ('rf', RandomForestClassifier(
             class_weight='balanced',  # cost-sensitive learning

            )
         )
    ])


def get_hyper_parameter_space():
    # https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use
    return {
        'anova__k': skopt.space.Integer(2, 14),
        'rf__max_depth': skopt.space.Categorical([3, 7, None]),
        'rf__n_estimators': skopt.space.Integer(5, 500),
        'rf__min_samples_split': skopt.space.Integer(2, 10),
        'rf__min_samples_leaf': skopt.space.Integer(1, 10),
    }
