import logging

from npfd.models.HTK import htktools as htkt

from ..paths import htk_misc_dir, hmm_model_path, interim_data_path, model_path, src_module_dir


class HiddenMarkovModel(object):
    def __init__(self):
        self.adapted = False
        self._params = None
        self.most_trained_model = None
        self.variance_floor = None

    def initialize(self, data, init_method=None, variance_floor=0, minimum_variance=0, **kwargs):#, **hyperparameters):
        """Initialize HMM
        # TODO: doc

        """
        if init_method is None:
            raise Exception('Initialization method is required')

        htkt.clean_models()

        logging.info("Initializing model...")
        logging.info(htk_misc_dir)
        if init_method == 'HCompV':
            htkt.HCompV(['-C', htk_misc_dir / 'config',
                         '-S', data['script_file'],
                         '-M', hmm_model_path / '0',
                         '-T', 1,
                         '-f', "{:.10f}".format(variance_floor),
                         '-v', "{:.10f}".format(minimum_variance),
                         '-m',
                         htk_misc_dir / 'proto'])

            htkt.gen_hmmdefs_from_proto()
            htkt.gen_macros()

        elif init_method == 'HInit':
            for label in ['ne', 'e']:
                htkt.HInit(['-C', htk_misc_dir / 'config',
                            '-S', data['script_file'],
                            '-I', '../data/interim/manual_labels_real_data.mlf',
                            '-M', hmm_model_path / '0',
                            # '-T', 1,
                            '-l', label,
                            '-o', label,
                            '-v', "{:.10f}".format(minimum_variance),
                            htk_misc_dir / 'proto'])

            htkt.HCompV(['-C', htk_misc_dir / 'config',
                         '-S', data['script_file'],
                         '-M', hmm_model_path / '0',
                         '-T', 1,
                         '-f', "{:.10f}".format(variance_floor),
                         '-m',
                         htk_misc_dir / 'proto'])

            htkt.gen_hmmdefs()
            htkt.gen_macros()

        self.most_trained_model = 0

        return 0

    def train(self, data, labels, **hyperparameters):
        logging.info("Training the model...")
        n = 3

        for i in range(self.most_trained_model, self.most_trained_model + n):
            htkt.HERest(['-C', htk_misc_dir / 'config',
                         '-I', labels['mlf'],
                         '-S', data['script_file'],
                         '-H', hmm_model_path / str(self.most_trained_model) / 'macros',
                         '-H', hmm_model_path / str(self.most_trained_model) / 'hmmdefs',
                         '-M', hmm_model_path / str(self.most_trained_model + 1),
                         '-s', hmm_model_path / str(self.most_trained_model + 1) / 'stats',
                         '-v', "{:0.10f}".format(hyperparameters['minimum_variance']),
                         # '-t', 250.0, 150.0, 1000.0,
                         # '-T', 1,
                         htk_misc_dir / 'monophones'])

            #model += 1
            self.most_trained_model += 1

        logging.info("Most trained model: " + str(self.most_trained_model))

        return self.most_trained_model

    def test(self, data, labels, out_mlf_file=None, **hyperparameters):
        logging.info("Testing model: " + str(self.most_trained_model))

        if out_mlf_file is None:
            out_mlf_file = 'results.mlf'

        if self.adapted:
            logging.info('Testing real data')
            htkt.HVite(['-C', htk_misc_dir / 'config',
                        '-H', hmm_model_path / str(self.most_trained_model) / 'macros',
                        '-H', hmm_model_path / str(self.most_trained_model) / 'hmmdefs',
                        '-p', "{:.10f}".format(hyperparameters['word_insertion_penalty']),
                        '-s', "{:.10f}".format(hyperparameters['grammar_scale_factor']),
                        '-A',
                        # '-T', 1,
                        '-J', model_path / 'classes',
                        '-J', model_path / 'xforms', 'mllr1',
                        '-h', '*/%*',
                        '-k',
                        '-w', htk_misc_dir / 'wdnet',
                        '-S', data['script_file'],
                        '-i', interim_data_path / out_mlf_file,
                        htk_misc_dir / 'dict',
                        htk_misc_dir / 'monophones'])

            r = htkt.HResults(['-I', labels['mlf'],
                               '-p',
                               htk_misc_dir / 'monophones',
                               interim_data_path / out_mlf_file])
        else:
            htkt.HVite(['-C', htk_misc_dir / 'config',
                        '-H', hmm_model_path / str(self.most_trained_model) / 'macros',
                        '-H', hmm_model_path / str(self.most_trained_model) / 'hmmdefs',
                        '-p', "{:.10f}".format(hyperparameters['word_insertion_penalty']),
                        '-s', "{:.10f}".format(hyperparameters['grammar_scale_factor']),
                        '-A',
                        # '-T', 1,
                        '-S', data['script_file'],
                        '-i', interim_data_path / out_mlf_file,
                        '-w', htk_misc_dir / 'wdnet',
                        htk_misc_dir / 'dict',
                        htk_misc_dir / 'monophones'])

            r = htkt.HResults(['-I', labels['mlf'],
                               '-p',
                               htk_misc_dir / 'monophones',
                               interim_data_path / out_mlf_file])

        r['mlf'] = interim_data_path / out_mlf_file

        return r

    def edit(self, commands, monophones_file=None):
        logging.info("Editing model " + str(self.most_trained_model))

        if monophones_file is None:
            monophones_file = htk_misc_dir / 'monophones'
            
        with open(src_module_dir / 'models/HTK/tmp/cmds.hed', 'wt') as cmd_file:
            for command in commands:
                cmd_file.write(command + '\n')

        htkt.HHEd(['-H', hmm_model_path / str(self.most_trained_model) / 'macros',
                   '-H', hmm_model_path / str(self.most_trained_model) / 'hmmdefs',
                   '-M', hmm_model_path / str(self.most_trained_model + 1),
                   '-T', 1,
                   src_module_dir / 'models/HTK/tmp/cmds.hed',
                   monophones_file])

        self.most_trained_model += 1

        logging.info("Most trained model: " + str(self.most_trained_model))

        return self.most_trained_model

    def adapt(self, data, labels,
              monophones_file=None):

        logging.info("Adapting model " + str(self.most_trained_model))

        if monophones_file is None:
            monophones_file = htk_misc_dir / 'monophones'
            
        htkt.HERest([
            '-C', htk_misc_dir / 'config',
            '-C', htk_misc_dir / 'config.globals',
            '-S', data['script_file'],
            '-I', labels['mlf'],
            '-H', hmm_model_path / str(self.most_trained_model) / 'macros',
            '-u', 'a',
            '-H', hmm_model_path / str(self.most_trained_model) / 'hmmdefs',
            # '-v', "{:.10f}".format(hyperparameters['minimum_variance']),
            # '-t', 250.0,
            '-T', 1,
            '-K', model_path / 'xforms', 'mllr1',
            '-J', model_path / 'classes',
            '-h', '*/%*',
            monophones_file])

        self.adapted = True

        return self.most_trained_model

    def predict(self, data, hyperparameters,
                config_file=None,
                wdnet_file=None,
                monophones_file=None,
                dict_file=None,
                out_mlf_file=None):
        """Invoke HVite
        
        """

        if out_mlf_file is None:
            out_mlf_file = 'results.mlf'

        if config_file is None:
            config_file = htk_misc_dir / 'config'
        
        if wdnet_file is None:
            wdnet_file = htk_misc_dir / 'wdnet'
            
        if monophones_file is None:
            monophones_file = htk_misc_dir / 'monophones'
            
        htkt.HVite(['-C', config_file,
                    '-H', hmm_model_path / str(self.most_trained_model) / 'macros',
                    '-H', hmm_model_path / str(self.most_trained_model) / 'hmmdefs',
                    #             '-p', 0,
                    #             '-s', 5,
                    '-A',
                    # '-T', 1,
                    '-p', "{:.10f}".format(hyperparameters['word_insertion_penalty']),
                    '-s', "{:.10f}".format(hyperparameters['grammar_scale_factor']),
                    '-J', hmm_model_path / 'classes',
                    '-J', hmm_model_path / 'xforms' / 'mllr1',
                    '-h', '*/%*',
                    '-k',
                    '-w', wdnet_file,
                    '-S', data['script_file'],
                    '-i', interim_data_path / out_mlf_file,
                    htk_misc_dir / 'dict',
                    monophones_file])

        return {'mlf': interim_data_path / out_mlf_file}
