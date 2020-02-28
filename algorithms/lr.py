from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import skopt


def get_classifier():
    # Mange ML metoder såsom Logistisk Regression, SVM, kNN, PCA kræver at features er skaleret
    scaler = MinMaxScaler()
    return Pipeline([
        ('scaler', scaler),
        ('anova', SelectKBest(f_classif, k=14)),
        ('lr', LogisticRegression(
             class_weight='balanced',  # cost-sensitive learning
             solver='liblinear',  # godt til lille datasæt og kan håndtere L1 regularisering
             max_iter=250,  # for at undgå `Liblinear failed to converge`
            )
         )
    ])


def get_hyper_parameter_space():
    return {
        'anova__k': skopt.space.Integer(2, 14),
        'lr__C': skopt.space.Real(0.001, 1000, prior='log-uniform'),
        'lr__penalty': skopt.space.Categorical(['l1', 'l2']),
    }
