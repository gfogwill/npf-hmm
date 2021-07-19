from npfd.models.base import HiddenMarkovModel
from npfd import data

from verta import Client

HOST = "http://127.0.0.1:3000/"

PROJECT_NAME = "NPF Detector"
EXPERIMENT_NAME = "Automated Test DMPS only (GP) - 03/07/2021"
EXPERIMENT_DESCRIPTION = "minimum variance disabled when adapting"

client = Client(HOST)
proj = client.set_project(PROJECT_NAME)
expt = client.set_experiment(EXPERIMENT_NAME)


def train_evaluate_with_adaptation(X_train, X_val, y_train, y_val, X_adapt, X_test, y_adapt, y_test,
                                   gaussian_duplication_times=6, **kwargs):
    run = client.set_experiment_run(desc=EXPERIMENT_DESCRIPTION)
    run.log_hyperparameters(kwargs)

    model = HiddenMarkovModel()
    model.initialize(X_train, init_method='HCompV', **kwargs)

    model.train(X_train, y_train, **kwargs)

    for i in range(1, gaussian_duplication_times + 1):
        model.edit([f'MU {2 ** i} {{*.state[2-4].mix}}'])
        model.train(X_train, y_train, **kwargs)

    model.adapt(X_adapt, y_adapt, **kwargs)

    results = model.test(X_test, y_test, **kwargs)

    try:
        score = results['WORD_Acc']
    except:
        score = -10
    if score < 0:
        score = -10

    run.log_metric('Accuracy', score)

    return results['WORD_Acc']


def train_evaluate(X_train, y_train, X_test, y_test, **kwargs):
    run = client.set_experiment_run(desc=EXPERIMENT_DESCRIPTION)
    run.log_hyperparameters(kwargs)

    model = HiddenMarkovModel()
    model.initialize(X_train, init_method='HCompV', **kwargs)

    model.train(X_train, y_train, **kwargs)

    for i in range(1, kwargs['gaussian_duplication_times'] + 1):
        model.edit([f'MU {2 ** i} {{*.state[2-4].mix}}'])
        model.train(X_train, y_train, **kwargs)

    results = model.test(X_test, y_test, **kwargs)

    run.log_metric('Accuracy', results['WORD_Acc'])

    return results['WORD_Acc']
