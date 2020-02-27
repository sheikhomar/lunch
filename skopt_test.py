def create(model, x, y, scoring = 'f1', n_folds: int = 5, random_state: int = 42, hyperparams_space=[]):
    if scoring == 'cohen_kappa':
        scoring = cohen_kappa_scorer
    def calc_scores(hyperparams=None):
        cv_splitter = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        if hyperparams is not None:
            model.set_params(**hyperparams)
        scores = cross_val_score(model, x, y, cv=cv_splitter, scoring=scoring, verbose=0, n_jobs=-1)
        return scores
    
    @skopt.utils.use_named_args(hyperparams_space)
    def objective(**hyperparams):
        scores = calc_scores(hyperparams)
        print('Hyperparams: {} max_score={:0.4f}'.format(hyperparams, scores.max()))
        return 1 - scores.max()

    def tune_hyperparams():
        res = skopt.gp_minimize(objective, hyperparams_space)
        return res
    
    return {
        'calc_scores': calc_scores,
        'objective': objective,
        'tune_hyperparams': tune_hyperparams
    }

def calc_cv_avg_score(model, x, y, scoring):
    return create(model, X_train, y_train, scoring)['calc_scores']().mean()

scaler = MinMaxScaler()  # Vigtigt for mange ML metoder såsom Logistisk Regression, SVM, kNN, PCA
lr_pipeline = Pipeline([
    ('scaler', scaler),
    ('anova', SelectKBest(f_classif, k=10)),
    ('lr',
         LogisticRegression(
            class_weight='balanced',  # cost-sensitive learning
            solver='liblinear',  # godt til lille datasæt og kan håndtere L1 regularisering
            max_iter=250,  # for at undgå `Liblinear failed to converge`
        )
    )
])
# calc_cv_avg_score(lr_pipeline, X_train, y_train, scoring='f1')

lr_hyperparam_space = [
    skopt.space.Real(0.001, 1000, name='lr__C', prior='log-uniform'),
    skopt.space.Categorical(['l1', 'l2'], name='lr__penalty'),
    skopt.space.Integer(2, 14, name='anova__k'),
]
tune_hyperparams = create(lr_pipeline, X_train, y_train, hyperparams_space=lr_hyperparam_space)['tune_hyperparams']
result = tune_hyperparams()