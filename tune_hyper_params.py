#!/usr/bin/env python
import argparse as ap
import importlib
import json
import os
from os import path
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV

ALGORITHMS_DIR = 'algorithms'

ALGORITHMS = [path.splitext(f)[0]
              for f in os.listdir(ALGORITHMS_DIR)
              if path.isfile(path.join(ALGORITHMS_DIR, f))]

TIME_DURATION_UNITS = (
    ('week', 60 * 60 * 24 * 7),
    ('day', 60 * 60 * 24),
    ('hour', 60 * 60),
    ('min', 60),
    ('sec', 1)
)


def import_algorithm(algorithm):
    if algorithm not in ALGORITHMS:
        msg = 'Unknown algorithm "%s"!' % algorithm
        raise ap.ArgumentTypeError(msg)
    return importlib.import_module('algorithms.%s' % algorithm)


def human_time_duration(seconds: float):
    # Source: https://gist.github.com/borgstrom/936ca741e885a1438c374824efb038b3
    if seconds is None or seconds <= 0:
        return 'inf'
    parts = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(int(seconds), div)
        if amount > 0:
            parts.append('{} {}{}'.format(amount, unit, '' if amount == 1 else 's'))
    return ', '.join(parts)


def save_results(searcher: BayesSearchCV, algorithm_name: str, duration_sec: float, args):
    n_folds = args.n_folds

    search_results = searcher.cv_results_
    num_params = len(search_results['params'])
    test_scores = np.zeros((n_folds, num_params))
    for i in range(n_folds):
        for j in range(num_params):
            test_scores[i][j] = search_results['split%d_test_score' % i][j]
    test_scores = test_scores.transpose()

    output = {
        'algorithm': algorithm_name,
        'execution_time': human_time_duration(duration_sec),
        'best_score': searcher.best_score_,
        'best_params': searcher.best_params_,
        'n_folds': n_folds,
        'n_iter': args.n_iter,
        'search_params': search_results['params'],
        'test_scores': test_scores.tolist(),
    }

    result_path = path.join('hyperparams', '{}.json'.format(algorithm_name))
    with open(result_path, 'w') as results_file:
        json.dump(output, results_file, sort_keys=False, indent=4, separators=(',', ': '))

    print('Results saved at {}'.format(result_path))


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=import_algorithm,
                        required=True,
                        help='Supported algorithms: %s' % ', '.join(ALGORITHMS))
    parser.add_argument('--n-folds', '-f',
                        type=int,
                        required=False,
                        default=5)
    parser.add_argument('--n-jobs', '-j',
                        type=int,
                        required=False,
                        default=-1)
    parser.add_argument('--n-iter', '-i',
                        type=int,
                        required=False,
                        default=200)
    return parser.parse_args()


def main():
    start_time = time()
    args = parse_args()

    print(args.__dict__)

    algorithm = args.algorithm
    algorithm_name = algorithm.__name__.replace('algorithms.', '')
    n_folds = args.n_folds
    n_jobs = args.n_jobs
    n_iter = args.n_iter
    target_feature = 'Churned'

    df_data = pd.read_csv('train_data.csv', index_col=0)
    y_train = df_data[target_feature]
    X_train = df_data.drop(target_feature, axis=1)

    classifier = algorithm.get_classifier()
    hyper_params_space = algorithm.get_hyper_parameter_space()

    print('Running hyper-parameter search...')
    cv_splitter = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    searcher = BayesSearchCV(classifier, hyper_params_space,
                             n_iter=n_iter, scoring='f1', verbose=1, n_jobs=n_jobs, cv=cv_splitter)
    searcher.fit(X_train, y_train)

    duration_sec = time() - start_time

    save_results(searcher, algorithm_name, duration_sec, args)


if __name__ == '__main__':
    main()
