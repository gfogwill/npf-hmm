import logging

from npfd.models.HTK import htktools as htkt


class HiddenMarkovModel(object):
    def __init__(self, **model_hyper_parameters):
        self._params = None
        self.most_trained_model = None

    def initialize(self, data, how='HCompV', variance_floor=0.1, minimum_variance=0):
        htkt.clean_models()

        logging.info("Initializing model...")

        if how == 'HCompV':
            htkt.HCompV(['-C', '../npfd/models/HTK/misc/config',
                         '-S', data['script_file'],
                         '-M', '../models/hmm/0',
                         '-T', 1,
                         '-f', variance_floor,
                         '-v', minimum_variance,
                         '-m',
                         '../npfd/models/HTK/misc/proto'])

            htkt.gen_hmmdefs_from_proto()
            htkt.gen_macros()

        elif how == 'HInit':
            for label in ['ne', 'e']:
                htkt.HInit(['-C', '../npfd/models/HTK/misc/config',
                            '-S', '../data/interim/train.scp',
                            '-I', '../data/interim/labels.mlf',
                            '-M', '../models/hmm/0',
                            # '-T', 1,
                            '-l', label,
                            '-o', label,
                            '-v', minimum_variance,
                            '../npfd/models/HTK/misc/proto'])

            htkt.HCompV(['-C', '../npfd/models/HTK/misc/config',
                         '-S', '../data/interim/train.scp',
                         '-M', '../models/hmm/0',
                         '-T', 1,
                         '-f', variance_floor,
                         '-m',
                         '../npfd/models/HTK/misc/proto'])

            htkt.gen_hmmdefs()
            htkt.gen_macros()

        self.most_trained_model = 0

        return 0

    def train(self, data, labels):
        n = 3

        for i in range(self.most_trained_model, self.most_trained_model + n):
            htkt.HERest(['-C', '../npfd/models/HTK/misc/config',
                         '-I', labels['mlf'],
                         '-S', data['script_file'],
                         '-H', '../models/hmm/' + str(self.most_trained_model) + '/macros',
                         '-H', '../models/hmm/' + str(self.most_trained_model) + '/hmmdefs',
                         '-M', '../models/hmm/' + str(self.most_trained_model + 1) + '/',
                         '-s', '../models/hmm/' + str(self.most_trained_model + 1) + '/stats',
                         # '-t', 250.0,# 150.0, 1000.0,
                         # '-T', 1,
                         '../npfd/models/HTK/misc/monophones'])

            #model += 1
            self.most_trained_model += 1

        return self.most_trained_model

    def test(self, data, labels):

        if data['id'].split('.')[2] == 'real':
            htkt.HVite(['-C', '../npfd/models/HTK/misc/config',
                        '-H', '../models/hmm/' + str(self.most_trained_model) + '/macros',
                        '-H', '../models/hmm/' + str(self.most_trained_model) + '/hmmdefs',
                        #             '-p', 0,
                        #             '-s', 5,
                        '-A',
                        # '-T', 1,
                        '-J', '../models/classes/',
                        '-J', '../models/xforms', 'mllr1',
                        '-h', '../data/interim/' + data['id'] + '/%*',
                        '-k',
                        '-w', '../npfd/models/HTK/misc/wdnet',
                        '-S', data['script_file'],
                        '-i', '../data/interim/results.mlf',
                        '../npfd/models/HTK/misc/dict',
                        '../npfd/models/HTK/misc/monophones'])

            r = htkt.HResults(['-I', labels['mlf'],
                               '-p',
                               '../npfd/models/HTK/misc/monophones',
                               '../data/interim/results.mlf'])
        else:
            htkt.HVite(['-C', '../npfd/models/HTK/misc/config',
                        '-H', '../models/hmm/' + str(self.most_trained_model) + '/macros',
                        '-H', '../models/hmm/' + str(self.most_trained_model) + '/hmmdefs',
                        '-p', 0,
                        '-s', 5,
                        '-A',
                        # '-T', 1,
                        '-S', data['script_file'],
                        '-i', '../data/interim/results.mlf',
                        '-w', '../npfd/models/HTK/misc/wdnet',
                        '../npfd/models/HTK/misc/dict',
                        '../npfd/models/HTK/misc/monophones'])

            r = htkt.HResults(['-I', labels['mlf'],
                               '-p',
                               '../npfd/models/HTK/misc/monophones',
                               '../data/interim/results.mlf'])

        r['mlf'] = '../data/interim/results.mlf'
        return r

    def edit(self, commands):

        cmd_file_path = '../npfd/models/HTK/tmp/cmds.hed'

        with open(cmd_file_path, 'wt') as cmd_file:
            for command in commands:
                cmd_file.write(command + '\n')

        htkt.HHEd(['-H', '../models/hmm/' + str(self.most_trained_model) + '/macros',
                   '-H', '../models/hmm/' + str(self.most_trained_model) + '/hmmdefs',
                   '-M', '../models/hmm/' + str(self.most_trained_model + 1) + '',
                   '-T', 1,
                   cmd_file_path,
                   '../npfd/models/HTK/misc/monophones'])

        #model += 1
        self.most_trained_model += 1

        return self.most_trained_model

    def adapt(self, data, labels):

        htkt.HERest([
            '-C', '../npfd/models/HTK/misc/config',
            '-C', '../npfd/models/HTK/misc/config.globals',
            '-S', data['script_file'],
            '-I', labels['mlf'],
            '-u', 'a',
            #     '-T',1,
            '-K', '../models/xforms', 'mllr1',
            '-J', '../models/classes/',
            '-h', '../data/interim/' + data['id'] + '/%*',
            '-H', '../models/hmm/' + str(self.most_trained_model) + '/macros',
            '-H', '../models/hmm/' + str(self.most_trained_model) + '/hmmdefs',
            '../npfd/models/HTK/misc/monophones'])

        # model += 1

        return self.most_trained_model

    def predict(self, data):
        htkt.HVite(['-C', '../npfd/models/HTK/misc/config',
                    '-H', '../models/hmm/' + str(self.most_trained_model) + '/macros',
                    '-H', '../models/hmm/' + str(self.most_trained_model) + '/hmmdefs',
                    #             '-p', 0,
                    #             '-s', 5,
                    '-A',
                    # '-T', 1,
                    '-J', '../models/classes/',
                    '-J', '../models/xforms', 'mllr1',
                    '-h', '../data/interim/' + data['id'] + '/%*',
                    '-k',
                    '-w', '../npfd/models/HTK/misc/wdnet',
                    '-S', data['script_file'],
                    '-i', '../data/interim/results.mlf',
                    '../npfd/models/HTK/misc/dict',
                    '../npfd/models/HTK/misc/monophones'])

        return {'mlf': '../data/interim/results.mlf'}