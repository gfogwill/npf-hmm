from npfd.models.base import HiddenMarkovModel
from npfd import data


SEARCH_PARAMS = {'normalize': True,
                 'data_version': '2',

                 # Labels
                 'label_type': 'event-noevent',
                 'nuc_threshold': 0.15,  # 1/cm3/10min
                 'pos_vol_threshold': 200,  # 1/m3^3/10min
                 'neg_vol_threshold': -5000,  # 1/cm3/10min

                 # Initialization
                 'variance_floor': 0.1,
                 'minimum_variance': 0.1,

                 'word_insertion_penalty': 0.0,
                 'grammar_scale_factor': 0.0
                 }

#
# from verta import Client
#
# HOST = "http://127.0.0.1:3000/"
#
# PROJECT_NAME = "NPF Detector"
# EXPERIMENT_NAME = "Automated Test - 25/01/2021"
# EXPERIMENT_DESCRIPTION = "minimum variance disabled when adapting"
#
#
# client = Client(HOST)
# proj = client.set_project(PROJECT_NAME)
# expt = client.set_experiment(EXPERIMENT_NAME)


def train_evaluate(search_params, X_train, X_val, y_train, y_val, X_adapt, X_test, y_adapt, y_test):
    # run = client.set_experiment_run(desc=EXPERIMENT_DESCRIPTION)

    params = {# Labels
              'init_metho': 'HCompV',
              # 'nuc_threshold': 0.15,  # 1/cm3/10min
              # 'pos_vol_threshold': 200,  # 1/m3^3/10min
              # 'neg_vol_threshold': -5000,  # 1/cm3/10min

              'word_insertion_penalty': 0.0,
              'grammar_scale_factor': 0.0,
              **search_params}

    edit_commands = ['MU 2 {*.state[2-4].mix}',
                     'MU 4 {*.state[2-4].mix}',
                     'MU 8 {*.state[2-4].mix}',
                     'MU 16 {*.state[2-4].mix}',
                     'MU 32 {*.state[2-4].mix}']#,
                     # 'MU 64 {*.state[2-4].mix}',
                     # 'MU 128 {*.state[2-4].mix}']
                     # 'MU 256 {*.state[2-4].mix}']

    # run.log_hyperparameters(params)
    model = HiddenMarkovModel()

    model.initialize(X_train, params)

    model.train(X_train, y_train, params)

    for cmd in edit_commands:
        model.edit([cmd])
        model.train(X_train, y_train, params)

    model.adapt(X_adapt, y_adapt)

    results = model.test(X_test, y_test, params)
    #TODO: make number of gaussians a parameter to be tuned.

    # results = model.test.synth(X_adapt, y_adapt)
    #
    # model.adapt(X_adapt, y_adapt, params)
    #
    # results = model.test(X_final, y_final, params)

    try:
        score = results['WORD_Acc']
    except KeyError:
        score = 0

    # run.log_metric('correct_labels', score)

    return score


if __name__ == '__main__':
    score = train_evaluate(SEARCH_PARAMS)
    print('validation AUC:', score)
