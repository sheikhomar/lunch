import skopt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def get_classifier():
    # Mange ML metoder såsom Logistisk Regression, SVM, kNN, PCA kræver at features er skaleret
    scaler = MinMaxScaler()
    return Pipeline([
        ('scaler', scaler),
        ('anova', SelectKBest(f_classif, k=14)),
        ('svm', SVC(
             class_weight='balanced',  # cost-sensitive learning
            )
         )
    ])


def get_hyper_parameter_space():
    return {
        'anova__k': skopt.space.Integer(2, 14),
        'svm__C': skopt.space.Real(0.0001, 1000, prior='log-uniform'),
        'svm__kernel': skopt.space.Categorical(['rbf', 'poly']),
        'svm__gamma': skopt.space.Categorical(['scale', 'auto']),
    }
