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


from verta import Client

HOST = "http://127.0.0.1:3000/"

PROJECT_NAME = "NPF Detector"
EXPERIMENT_NAME = "Automated Test - 25/01/2021"
EXPERIMENT_DESCRIPTION = "minimum variance disabled when adapting"


client = Client(HOST)
proj = client.set_project(PROJECT_NAME)
expt = client.set_experiment(EXPERIMENT_NAME)


def train_evaluate(search_params):

    run = client.set_experiment_run(desc=EXPERIMENT_DESCRIPTION)

    params = {'init_metho': 'HCompV',
              'raw_data_source': 'simulation',
              'normalize': True,
              'data_version': '2',

              # Labels
              'label_type': 'event-noevent',
              'nuc_threshold': 0.15,  # 1/cm3/10min
              'pos_vol_threshold': 200,  # 1/m3^3/10min
              'neg_vol_threshold': -5000,  # 1/cm3/10min
              'word_insertion_penalty': 0.0,
              'grammar_scale_factor': 0.0,
              **search_params}

    run.log_hyperparameters(params)

    X_train, X_val, y_train, y_val = data.dataset.make_dataset(params, clean_interim_dir=False)

    params['raw_data_source'] = 'real'
    X_adapt, X_final, y_adapt, y_final = data.dataset.make_dataset(params, test_size=0.2)

    model = HiddenMarkovModel()

    model.initialize(X_train, params)

    model.train(X_train, y_train, params)
    # results = model.test.synth(X_val, y_val)

    edit_commands = ['MU 2 {*.state[2-4].mix}']

    model.edit(edit_commands)
    model.train(X_train, y_train, params)

    edit_commands = ['MU 4 {*.state[2-4].mix}']

    model.edit(edit_commands)
    model.train(X_train, y_train, params)

    edit_commands = ['MU 8 {*.state[2-4].mix}']

    model.edit(edit_commands)
    model.train(X_train, y_train, params)

    # results = model.test.synth(X_adapt, y_adapt)

    model.adapt(X_adapt, y_adapt, params)

    results = model.test(X_final, y_final, params)

    try:
        score = results['WORD_Acc']
    except KeyError:
        score = 0

    run.log_metric('correct_labels', score)

    return score


if __name__ == '__main__':
    score = train_evaluate(SEARCH_PARAMS)
    print('validation AUC:', score)
