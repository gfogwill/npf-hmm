"""
    HTKtools
    ~~~
    This module contains functions implemented in HTK.

    :copyright: (c) 2012 by Scott Davies for Web Drive Ltd.
"""

import os
import shutil
import subprocess

import logging

logging.basicConfig(level=logging.INFO)

HTKDIR = '/home/gfogwil/Documentos/Facultad/Tesis/programs/htk/HTKTools/'
NUMBER_OF_SIZE_BINS = 25

HTKTOOLDIR = os.path.dirname(__file__)
MONOPHONES_PATH = os.path.join(HTKTOOLDIR, './misc/monophones')
PROTO_PATH = os.path.join(HTKTOOLDIR, './misc/proto')

OUT_INIT_PROTO_PATH = os.path.join(HTKTOOLDIR, '../../../models/hmm/0/proto')
OUT_INIT_HMMDEFS_PATH = os.path.join(HTKTOOLDIR, '../../../models/hmm/0/hmmdefs')
OUT_INIT_VFLOORS_PATH = os.path.join(HTKTOOLDIR, '../../../models/hmm/0/vFloors')
OUT_INIT_MACROS_PATH = os.path.join(HTKTOOLDIR, '../../../models/hmm/0/macros')

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../../models/hmm/')


def clean_models():
    clean_dir(MODELS_DIR)

    for i in range(50):
        os.mkdir(MODELS_DIR + str(i))


def gen_hmmdefs_from_proto(monophones=MONOPHONES_PATH,
                           proto=OUT_INIT_PROTO_PATH,
                           output=OUT_INIT_HMMDEFS_PATH):
    print(os.getcwd())

    with open(proto) as proto_file:
        proto_str = proto_file.readlines()[3:]

        with open(monophones) as monophones_file:
            monophones_list = monophones_file.read().split()

        with open(output, 'wt') as hmmdefs:
            for mono in monophones_list:
                hmmdefs.write(proto_str[0].replace('proto', mono))
                [hmmdefs.write(line) for line in proto_str[1:]]


def gen_hmmdefs(output=OUT_INIT_HMMDEFS_PATH):

    with open(MODELS_DIR + '0/ne') as ne_file:
        ne_str = ne_file.readlines()[3:]
    with open(MODELS_DIR + '0/e') as e_file:
        e_str = e_file.readlines()[3:]

    with open(output, 'wt') as hmmdefs:
        [hmmdefs.write(line) for line in ne_str]
        [hmmdefs.write(line) for line in e_str]


def gen_macros(vFloors=OUT_INIT_VFLOORS_PATH,
               proto=PROTO_PATH,
               output=OUT_INIT_MACROS_PATH):

    with open(output, 'wt') as out_file:
        with open(proto) as proto_file:
            out_file.write(proto_file.readline())

        with open(vFloors) as vFloors_file:
            for line in vFloors_file:
                out_file.write(line)


def display_artifact(which='all'):
    if which is 'all':
        for file in os.listdir('../npfd/HTK/misc'):
            print("****   npfd/models/HTK/misc/" + file + "   ****\n")
            with open(os.path.join('../npfd/models/HTK/misc', file)) as fi:
                print(fi.read())

    else:
        for artifact in which:
            print("****   npfd/HTK/misc/" + artifact + "   ****\n")
            with open(os.path.join('../npfd/HTK/misc', artifact)) as fi:
                print(fi.read())


def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def HCopy(args, print_output=False):
    """ Runs HTK HCopy program to calculate signal parameters.
        Args:
            args: Same arguments implemented in HTK
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """

    command = [HTKDIR + "HCopy"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return output.split().__len__() / 3


def HERest(args, print_output=True):
    """ Runs HTK HERest program to calculate signal parameters.
        Args:
            args (any): Path to configuration file.
            print_output (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """

    command = [HTKDIR + "HERest"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return


def HEAdapt(args, print_output=True):
    """ Runs HTK HEAdapt program to calculate signal parameters.
        Args:
            args (any): Path to configuration file.
            print_output (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """

    command = [HTKDIR + "HEAdapt"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return


def HRest(args, print_output=True):
    """ Runs HTK HERest program to calculate signal parameters.
        Args:
            args (string): Path to configuration file.
            print_output (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """

    command = [HTKDIR + "HRest"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return


def HVite(args, print_output=True):
    """ Runs HTK HERest program to calculate signal parameters.
        Args:
            args (any): Path to configuration file.
            print_output (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """

    command = [HTKDIR + "HVite"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return


def HInit(args, print_output=True):
    """ Runs HTK HERest program to calculate signal parameters.
        Args:
            as (string): Path to configuration file.
            print_output (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """

    command = [HTKDIR + "HInit"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return


def HCompV(args, print_output=True):
    """ Runs HTK HCompV program to initialize de model.
        Args:
            conf (string): Path to configuration file.
            script_file (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """
    command = [HTKDIR + "HCompV"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return


def HHEd(args, print_output=True):
    """ Runs HTK HHEd program to initialize de model.
        Args:
            conf (string): Path to configuration file.
            script_file (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """
    command = [HTKDIR + "HHEd"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    return


def HResults(args, print_output=True):
    """ Runs HTK HResults program to initialize de model.
        Args:
            conf (string): Path to configuration file.
            script_file (string): Path to script file to be used by HCopy.
        Returns:
            string: Program output and possible errors. None if program didn't run.
    """
    command = [HTKDIR + "HResults"]
    command.extend(args)
    args_str = [str(x) for x in command]

    try:
        output = subprocess.check_output(args_str, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(*e.cmd, sep=' ')
        [print(line) for line in e.output.decode().splitlines()]
        return

    if print_output:
        print(output.decode())

    results = {}

    for line in output.decode().split(sep='\n'):
        if line.startswith('SENT'):
            tmp = line.replace('SENT:', '').replace('[', '').replace(']', '').replace(',', '').replace('%', '').split()
            for r in tmp:
                results['SENT_' + r.split('=')[0]] = float(r.split('=')[1])
        elif line.startswith('WORD'):
            tmp = line.replace('WORD:', '').replace('[', '').replace(']', '').replace(',', '').replace('%', '').split()
            for r in tmp:
                results['WORD_' + r.split('=')[0]] = float(r.split('=')[1])
    return results


if __name__ == '__main__':
    model = 7
    HCopy(['-S', '../data/interim/real_files.scp',
                  '-I', '../data/raw/dmps/manual_labels.mlf',
                  '-p', 0.000001,
                  '-H', '../models/hmm/' + str(model) + 'macros',
                  '-H', '../models/hmm/' + str(model) + 'hmmdefs',
                  '-M', '../models/hmm/' + str(model + 1)])