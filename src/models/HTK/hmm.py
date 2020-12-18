from src.models.HTK import htktools as htkt

import logging


def initialize(data, how='HCompV', variance_floor=0.1, minimum_variance=0):
    htkt.clean_models()

    logging.info("Initializing model...")

    if how == 'HCompV':
        htkt.HCompV(['-C', '../src/models/HTK/misc/config',
                     '-S', data['script_file'],
                     '-M', '../models/hmm/0',
                     '-T', 1,
                     '-f', variance_floor,
                     '-v', minimum_variance,
                     '-m',
                     '../src/models/HTK/misc/proto'])

        htkt.gen_hmmdefs_from_proto()
        htkt.gen_macros()

    elif how == 'HInit':
        for label in ['ne', 'e']:
            htkt.HInit(['-C', '../src/models/HTK/misc/config',
                        '-S', '../data/interim/train.scp',
                        '-I', '../data/interim/labels.mlf',
                        '-M', '../models/hmm/0',
                        #'-T', 1,
                        '-l', label,
                        '-o', label,
                        '-v', minimum_variance,
                        '../src/models/HTK/misc/proto'])

        htkt.HCompV(['-C', '../src/models/HTK/misc/config',
                    '-S', '../data/interim/train.scp',
                     '-M', '../models/hmm/0',
                     '-T', 1,
                     '-f', variance_floor,
                     '-m',
                     '../src/models/HTK/misc/proto'])

        htkt.gen_hmmdefs()
        htkt.gen_macros()

    return 0


def train(model, data, labels):
    n = 3

    for i in range(model, model + n):
        htkt.HERest(['-C', '../src/models/HTK/misc/config',
                     '-I', labels['mlf'],
                     '-S', data['script_file'],
                     '-H', '../models/hmm/' + str(model) + '/macros',
                     '-H', '../models/hmm/' + str(model) + '/hmmdefs',
                     '-M', '../models/hmm/' + str(model + 1) + '/',
                     '-s', '../models/hmm/' + str(model + 1) + '/stats',
                     # '-t', 250.0,# 150.0, 1000.0,
                     # '-T', 1,
                     '../src/models/HTK/misc/monophones'])

        model += 1

    return model


def test(model, data, labels):

    if data['id'].split('.')[2] == 'real':
        htkt.HVite(['-C', '../src/models/HTK/misc/config',
                    '-H', '../models/hmm/' + str(model) + '/macros',
                    '-H', '../models/hmm/' + str(model) + '/hmmdefs',
                    #             '-p', 0,
                    #             '-s', 5,
                    '-A',
                    # '-T', 1,
                    '-J', '../models/classes/',
                    '-J', '../models/xforms', 'mllr1',
                    '-h', '../data/interim/' + data['id'] +'/%*',
                    '-k',
                    '-w', '../src/models/HTK/misc/wdnet',
                    '-S', data['script_file'],
                    '-i', '../data/interim/results.mlf',
                    '../src/models/HTK/misc/dict',
                    '../src/models/HTK/misc/monophones'])

        r = htkt.HResults(['-I', labels['mlf'],
                           '-p',
                           '../src/models/HTK/misc/monophones',
                           '../data/interim/results.mlf'])
    else:
        htkt.HVite(['-C', '../src/models/HTK/misc/config',
                    '-H', '../models/hmm/' + str(model) + '/macros',
                    '-H', '../models/hmm/' + str(model) + '/hmmdefs',
                    '-p', 0,
                    '-s', 5,
                    '-A',
                    # '-T', 1,
                    '-S', data['script_file'],
                    '-i', '../data/interim/results.mlf',
                    '-w', '../src/models/HTK/misc/wdnet',
                    '../src/models/HTK/misc/dict',
                    '../src/models/HTK/misc/monophones'])

        r = htkt.HResults(['-I', labels['mlf'],
                           '-p',
                           '../src/models/HTK/misc/monophones',
                           '../data/interim/results.mlf'])

    r['mlf'] = '../data/interim/results.mlf'
    return r


def edit(model, commands):

    cmd_file_path = '../src/models/HTK/tmp/cmds.hed'

    with open(cmd_file_path, 'wt') as cmd_file:
        for command in commands:
            cmd_file.write(command + '\n')

    htkt.HHEd(['-H', '../models/hmm/' + str(model) + '/macros',
               '-H', '../models/hmm/' + str(model) + '/hmmdefs',
               '-M', '../models/hmm/' + str(model + 1) + '',
               '-T', 1,
               cmd_file_path,
               '../src/models/HTK/misc/monophones'])

    model += 1

    return model


def adapt(model, data, labels):

    htkt.HERest([
        '-C', '../src/models/HTK/misc/config',
        '-C', '../src/models/HTK/misc/config.globals',
        '-S', data['script_file'],
        '-I', labels['mlf'],
        '-u', 'a',
        #     '-T',1,
        '-K', '../models/xforms', 'mllr1',
        '-J', '../models/classes/',
        '-h', '../data/interim/' + data['id'] +'/%*',
        '-H', '../models/hmm/' + str(model) + '/macros',
        '-H', '../models/hmm/' + str(model) + '/hmmdefs',
        '../src/models/HTK/misc/monophones'])

    # model += 1

    return model


def predict(model, data):
    htkt.HVite(['-C', '../src/models/HTK/misc/config',
                '-H', '../models/hmm/' + str(model) + '/macros',
                '-H', '../models/hmm/' + str(model) + '/hmmdefs',
                #             '-p', 0,
                #             '-s', 5,
                '-A',
                # '-T', 1,
                '-J', '../models/classes/',
                '-J', '../models/xforms', 'mllr1',
                '-h', '../data/interim/' + data['id'] + '/%*',
                '-k',
                '-w', '../src/models/HTK/misc/wdnet',
                '-S', data['script_file'],
                '-i', '../data/interim/results.mlf',
                '../src/models/HTK/misc/dict',
                '../src/models/HTK/misc/monophones'])

    return {'mlf': '../data/interim/results.mlf'}
