# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np

from src.models.HTK.htktools import HCopy, clean_dir
from src.data.size_distribution import cm3_to_dndlogdp
from src.data.htk import write_data


def make_dataset(normalize=True, data_version=2):
    """ Generate data files to be used by HTK

    """
    data_version = str(data_version)
    # Remove all files from interim directory
    clean_interim()

    logging.info('Converting raw files to HTK format ...')
    logging.info('Data version: ' + data_version)

    # Convert and split file into test and train
    for file in os.listdir('../data/raw/simulation/'):
        if file.endswith('.h5') and file.startswith(data_version):
            file_name = file[:-3]

            # Read in size distribution data
            size_dist_df = pd.read_hdf('../data/raw/simulation/' + file, key='obs/particle')

            # Resample data to 10 minutes
            size_dist_df = size_dist_df.resample('10T').mean()

            if normalize:
                # size_dist_df = pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df / 1e6))
                size_dist_df = np.log10(
                    np.absolute(pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df / 1e6)) + 10))
            else:
                size_dist_df = size_dist_df / 1e6
            # print(size_dist_df.describe())
            # Split data, 90% for training, 10% for testing
            if np.random.rand() < 0.1:
                fo = os.path.join('../data/interim/test', file_name[2:])
            else:
                fo = os.path.join('../data/interim/train', file_name[2:])

            # Write data
            write_data(fo, size_dist_df)

    train_file, test_file = gen_scp_files()

    logging.info('Adding deltas and acelerations...')

    test_count = HCopy(['-C', '../src/models/HTK/misc/config.hcopy',
                        '-S', '../data/interim/test_hcopy.scp',
                        '-T', 1])
    logging.info("Test files:\t" + str(test_count))

    # Train
    train_count = HCopy(['-C', '../src/models/HTK/misc/config.hcopy',
                         '-S', '../data/interim/train_hcopy.scp',
                         '-T', 1])
    logging.info("Train files:\t" + str(train_count))

    train = {'script_file': train_file, 'count': train_count, 'id': data_version + '.train.synth'}
    test = {'script_file': test_file, 'count': test_count, 'id': data_version + '.test.synth'}

    return train, test


def clean_interim():
    clean_dir('../data/interim/')
    clean_dir('../reports/figures/')

    os.mkdir('../data/interim/labels')
    os.mkdir('../data/interim/results')
    os.mkdir('../data/interim/test')
    os.mkdir('../data/interim/train')
    os.mkdir('../data/interim/test_D_A')
    os.mkdir('../data/interim/train_D_A')


def gen_scp_files():
    test_dir = '../data/interim/test'
    test_D_A_dir = '../data/interim/test_D_A'
    train_dir = '../data/interim/train'
    train_D_A_dir = '../data/interim/train_D_A'

    logging.info('Generating script (.scp) files...')

    with open('../data/interim/train_hcopy.scp', 'wt') as fi:
        for file in os.listdir(train_dir):
            line = os.path.join(train_dir, file) + ' ' + os.path.join(train_D_A_dir, file) + '\n'
            logging.debug(line)
            fi.write(line)

    with open('../data/interim/train.scp', 'wt') as fi:
        for file in os.listdir(train_dir):
            line = os.path.join(train_D_A_dir, file) + '\n'
            fi.write(line)

    with open('../data/interim/test_hcopy.scp', 'wt') as fi:
        for file in os.listdir(test_dir):
            line = os.path.join(test_dir, file) + ' ' + os.path.join(test_D_A_dir, file) + '\n'
            fi.write(line)

    with open('../data/interim/test.scp', 'wt') as fi:
        for file in os.listdir(test_dir):
            line = os.path.join(test_D_A_dir, file) + '\n'
            fi.write(line)

    return '../data/interim/train.scp', '../data/interim/test.scp'
