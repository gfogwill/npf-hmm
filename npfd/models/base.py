import logging
import os

from npfd.models.HTK import htktools as htkt

COMMANDS_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../npfd/models/HTK/tmp/cmds.hed')

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models/hmm/')
XFORMS_DIR = os.path.join(os.path.dirname(__file__), '../../models/xforms/')
CLASSES_DIR = os.path.join(os.path.dirname(__file__), '../../models/classes/')

RESULTS_MLF_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/results.mlf')
INTERIM_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/interim/')
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), './HTK/misc/config')
CONFIG_GLOBALS_FILE_PATH = os.path.join(os.path.dirname(__file__), './HTK/misc/config.globals')
MONOPHONES_PATH = os.path.join(os.path.dirname(__file__), './HTK/misc/monophones')
PROTO_PATH = os.path.join(os.path.dirname(__file__), './HTK/misc/proto')
DICT_PATH = os.path.join(os.path.dirname(__file__), './HTK/misc/dict')
WDNET_PATH = os.path.join(os.path.dirname(__file__), './HTK/misc/wdnet')


class HiddenMarkovModel(object):
    def __init__(self, **model_hyper_parameters):
        self.adapted = False
        self._params = None
        self.most_trained_model = None

    def initialize(self, data, hyperparameters):
        htkt.clean_models()

        logging.info("Initializing model...")

        if hyperparameters['init_metho'] == 'HCompV':
            htkt.HCompV(['-C', CONFIG_FILE_PATH,
                         '-S', data['script_file'],
                         '-M', MODELS_DIR + '0',
                         '-T', 1,
                         '-f', "{:.10f}".format(hyperparameters['variance_floor']),
                         '-v', "{:.10f}".format(hyperparameters['minimum_variance']),
                         '-m',
                         PROTO_PATH])

            htkt.gen_hmmdefs_from_proto()
            htkt.gen_macros()

        elif hyperparameters['init_metho'] == 'HInit':
            for label in ['ne', 'e']:
                htkt.HInit(['-C', CONFIG_FILE_PATH,
                            '-S', data['script_file'],
                            '-I', '../data/interim/manual_labels_real_data.mlf',
                            '-M', MODELS_DIR + '0',
                            # '-T', 1,
                            '-l', label,
                            '-o', label,
                            '-v', "{:.10f}".format(hyperparameters['minimum_variance']),
                            PROTO_PATH])

            htkt.HCompV(['-C', CONFIG_FILE_PATH,
                         '-S', data['script_file'],
                         '-M', MODELS_DIR + '0',
                         '-T', 1,
                         '-f', "{:.10f}".format(hyperparameters['variance_floor']),
                         '-m',
                         PROTO_PATH])

            htkt.gen_hmmdefs()
            htkt.gen_macros()

        self.most_trained_model = 0

        return 0

    def train(self, data, labels, hyperparameters):
        logging.info("Training the model...")
        n = 3

        for i in range(self.most_trained_model, self.most_trained_model + n):
            htkt.HERest(['-C', CONFIG_FILE_PATH,
                         '-I', labels['mlf'],
                         '-S', data['script_file'],
                         '-H', MODELS_DIR + str(self.most_trained_model) + '/macros',
                         '-H', MODELS_DIR + str(self.most_trained_model) + '/hmmdefs',
                         '-M', MODELS_DIR + str(self.most_trained_model + 1) + '/',
                         '-s', MODELS_DIR + str(self.most_trained_model + 1) + '/stats',
                         '-v', "{:0.10f}".format(hyperparameters['minimum_variance']),
                         # '-t', 250.0,# 150.0, 1000.0,
                         # '-T', 1,
                         MONOPHONES_PATH])

            #model += 1
            self.most_trained_model += 1

        logging.info("Most trained model: " + str(self.most_trained_model))

        return self.most_trained_model

    def test(self, data, labels, hyperparameters):
        logging.info("Testing model: " + str(self.most_trained_model))

        if self.adapted:
            logging.info('Testing real data')
            htkt.HVite(['-C', CONFIG_FILE_PATH,
                        '-H', MODELS_DIR + str(self.most_trained_model) + '/macros',
                        '-H', MODELS_DIR + str(self.most_trained_model) + '/hmmdefs',
                        '-p', "{:.10f}".format(hyperparameters['word_insertion_penalty']),
                        '-s', "{:.10f}".format(hyperparameters['grammar_scale_factor']),
                        '-A',
                        # '-T', 1,
                        '-J', CLASSES_DIR,
                        '-J', XFORMS_DIR, 'mllr1',
                        '-h', '*/%*',
                        '-k',
                        '-w', WDNET_PATH,
                        '-S', data['script_file'],
                        '-i', RESULTS_MLF_PATH,
                        DICT_PATH,
                        MONOPHONES_PATH])

            r = htkt.HResults(['-I', labels['mlf'],
                               '-p',
                               MONOPHONES_PATH,
                               RESULTS_MLF_PATH])
        else:
            htkt.HVite(['-C', CONFIG_FILE_PATH,
                        '-H', MODELS_DIR + str(self.most_trained_model) + '/macros',
                        '-H', MODELS_DIR + str(self.most_trained_model) + '/hmmdefs',
                        '-p', "{:.10f}".format(hyperparameters['word_insertion_penalty']),
                        '-s', "{:.10f}".format(hyperparameters['grammar_scale_factor']),
                        '-A',
                        # '-T', 1,
                        '-S', data['script_file'],
                        '-i', RESULTS_MLF_PATH,
                        '-w', WDNET_PATH,
                        DICT_PATH,
                        MONOPHONES_PATH])

            r = htkt.HResults(['-I', labels['mlf'],
                               '-p',
                               MONOPHONES_PATH,
                               RESULTS_MLF_PATH])

        r['mlf'] = RESULTS_MLF_PATH
        return r

    def edit(self, commands):
        logging.info("Editing model " + str(self.most_trained_model))

        with open(COMMANDS_FILE_PATH, 'wt') as cmd_file:
            for command in commands:
                cmd_file.write(command + '\n')

        htkt.HHEd(['-H', MODELS_DIR + str(self.most_trained_model) + '/macros',
                   '-H', MODELS_DIR + str(self.most_trained_model) + '/hmmdefs',
                   '-M', MODELS_DIR + str(self.most_trained_model + 1) + '',
                   '-T', 1,
                   COMMANDS_FILE_PATH,
                   MONOPHONES_PATH])

        self.most_trained_model += 1

        logging.info("Most trained model: " + str(self.most_trained_model))

        return self.most_trained_model

    def adapt(self, data, labels, hyperparameters):
        logging.info("Adapting model " + str(self.most_trained_model))

        print(MODELS_DIR)
        print(self.most_trained_model)

        htkt.HERest([
            '-C', CONFIG_FILE_PATH,
            '-C', CONFIG_GLOBALS_FILE_PATH,
            '-S', data['script_file'],
            '-I', labels['mlf'],
            '-u', 'a',
            # '-v', "{:.10f}".format(hyperparameters['minimum_variance']),
            # '-t', 250.0,
            #     '-T',1,
            '-K', XFORMS_DIR, 'mllr1',
            '-J', CLASSES_DIR,
            # '-h', INTERIM_DATA_DIR + data['id'] + '/%*',
            '-h', '*/%*',
            '-H', MODELS_DIR + str(self.most_trained_model) + '/macros',
            '-H', MODELS_DIR + str(self.most_trained_model) + '/hmmdefs',
            MONOPHONES_PATH])

        self.adapted = True

        return self.most_trained_model

    def predict(self, data, hyperparameters):
        htkt.HVite(['-C', CONFIG_FILE_PATH,
                    '-H', MODELS_DIR + str(self.most_trained_model) + '/macros',
                    '-H', MODELS_DIR + str(self.most_trained_model) + '/hmmdefs',
                    #             '-p', 0,
                    #             '-s', 5,
                    '-A',
                    # '-T', 1,
                    '-p', "{:.10f}".format(hyperparameters['word_insertion_penalty']),
                    '-s', "{:.10f}".format(hyperparameters['grammar_scale_factor']),
                    '-J', CLASSES_DIR,
                    '-J', XFORMS_DIR, 'mllr1',
                    '-h', '*/%*',
                    '-k',
                    '-w', WDNET_PATH,
                    '-S', data['script_file'],
                    '-i', RESULTS_MLF_PATH,
                    DICT_PATH,
                    MONOPHONES_PATH])

        return {'mlf': RESULTS_MLF_PATH}
