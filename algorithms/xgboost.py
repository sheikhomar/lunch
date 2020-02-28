import skopt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import xgboost as xgb


def get_classifier():
    return Pipeline([
        ('anova', SelectKBest(f_classif, k=14)),
        ('xgb', xgb.XGBClassifier(

            )
         )
    ])


def get_hyper_parameter_space():
    # https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
    return {
        'anova__k': skopt.space.Integer(2, 14),
        'xgb__max_depth': skopt.space.Categorical([2, 3, 4, 5, 6, 7, 12, None]),
        'xgb__min_child_weight': skopt.space.Integer(1, 10),
        'xgb__gamma': skopt.space.Real(0, 100),
    }
