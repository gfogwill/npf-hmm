# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import datetime

from npfd.models.HTK.htktools import HCopy, clean_dir
from npfd.data.size_distribution import cm3_to_dndlogdp, decimalDOY2datetime
from npfd.data.htk import write_data
from npfd.data.labels import get_labels_ene, get_labels_nccd, write_label, master_label_file, dmps_master_label_file

from ..paths import raw_data_path, interim_data_path, figures_path, htk_misc_dir

DMPS_TEST_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/dmps/dmps_mbio_2015/DATA/')

RAW_SIMULATION_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/malte-uhma/')

ADAPT_FILES = ['20170115.cle',
               '20170204.cle',
               '20170324.cle',
               '20171117.cle',
               '20171225.cle',
               '20170220.cle',
               '20170130.cle',
               '20171202.cle',
               ]


def make_dataset(dataset_name=None, clean_interim_dir=False, test_size=0.1):
    r"""Generates data files

    Long description

    Parameters
    ----------
    dataset_name : {'malte-uhma', 'real'}, optional
        Choices in brackets, default first when optional.
    *args : iterable
        Other arguments.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    X :
    y :

    """

    # Remove all files from interim directory
    if clean_interim_dir:
        clean_interim()

    if dataset_name == 'malte-uhma':
        logging.info('Converting malte-uhma raw files to HTK format ...')
        X_train, X_test, y_train, y_test = read_raw_simulations(test_size)
    elif dataset_name == 'dmps':
        logging.info('Converting real raw files to HTK format ...')
        X_train, X_test, y_train, y_test = read_raw_dmps(test_size=test_size)
    else:
        raise ValueError("Invalid option: " + dataset_name)

    return X_train, X_test, y_train, y_test


def read_raw_dmps(skip_invalid_day=False, clean_existing_data=True, test_size=0.1):
    dataset_name = 'dmps'

    train_data_path = interim_data_path / f'{dataset_name}_train'
    test_data_path = interim_data_path / f'{dataset_name}_test'
    train_D_A_data_path = interim_data_path / f'{dataset_name}_train_D_A'
    test_D_A_data_path = interim_data_path / f'{dataset_name}_test_D_A'
    train_labels_path = interim_data_path / f'{dataset_name}_labels_train'
    test_labels_path = interim_data_path / f'{dataset_name}_labels_test'

    train_scp_file = interim_data_path / f'{dataset_name}_train_D_A.scp'
    test_scp_file = interim_data_path / f'{dataset_name}_test_D_A.scp'

    try:
        os.mkdir(train_data_path)
        os.mkdir(test_data_path)
        os.mkdir(train_D_A_data_path)
        os.mkdir(test_D_A_data_path)
        os.mkdir(train_labels_path)
        os.mkdir(test_labels_path)
    except FileExistsError:
        # pass
        if clean_existing_data:
            clean_dir(train_data_path)
            clean_dir(test_data_path)
            clean_dir(train_D_A_data_path)
            clean_dir(test_D_A_data_path)
            clean_dir(train_labels_path)
            clean_dir(test_labels_path)
        else:
            pass

    labels = pd.read_csv(raw_data_path / 'dmps' / 'event_classification.csv', index_col=0)
    labels.index = pd.to_datetime(labels.index)
    count = 0

    for file in (raw_data_path / 'dmps/inv').glob('*.cle'):
        nukdata = pd.read_csv(raw_data_path / 'dmps/inv' / file, sep=r'\s+')
        nukdata = nukdata.replace(np.nan, -999)
        nukdata.index = nukdata.iloc[:, 0].apply(decimalDOY2datetime)

        # TODO: change ffill
        # nukdata = nukdata.drop(columns=nukdata.columns[[0, 1]]).resample('10T').ffill()
        nukdata = nukdata.drop(columns=nukdata.columns[[0, 1]]).resample('10T').mean()
        nukdata = nukdata.replace(np.nan, -999)

        if skip_invalid_day and nukdata.isin([-999]).any().any():
            continue

        # Split data, 90% for training, 10% for testing
        if np.random.rand() < test_size:
            fo = test_data_path / file.stem[2:]
            fo_label = test_labels_path / file.stem[2:]
            fo_D_A = test_D_A_data_path / file.stem[2:]
        else:
            fo = train_data_path / file.stem[2:]
            fo_label = train_labels_path / file.stem[2:]
            fo_D_A = train_D_A_data_path / file.stem[2:]

        day_labels = labels.loc[nukdata.index[0]:nukdata.index[0]+datetime.timedelta(days=1)]

        day_labels.where(nukdata.min(axis=1) != -999, 'na', inplace=True)

        # # Write labels to file
        write_label(fo_label, day_labels)

        write_data(fo, np.log10(np.absolute(nukdata + 10)))

        HCopy([fo, fo_D_A, '-C', htk_misc_dir / 'config.hcopy'])

        # scp_file.write(fo_D_A + '\n')
        count += 1

    train_labels, test_labels = dmps_master_label_file()

    with open(train_scp_file, 'wt') as fi:
        for file in os.listdir(os.path.join(interim_data_path, 'train_D_A.real')):
            line = os.path.join(os.path.join(interim_data_path, 'train_D_A.real'), file) + '\n'
            fi.write(line)

    with open(test_scp_file, 'wt') as fi:
        for file in os.listdir(os.path.join(interim_data_path, 'test_D_A.real')):
            line = os.path.join(os.path.join(interim_data_path, 'test_D_A.real'), file) + '\n'
            fi.write(line)

    X_train = {'script_file': train_scp_file, 'count': 1, 'id': 'train_D_A.real'}
    X_test = {'script_file': test_scp_file, 'count': 1, 'id': 'test_D_A.real'}

    y_train = {'mlf': train_labels, 'count': None, 'id': 'train.real'}
    y_test = {'mlf': test_labels, 'count': None, 'id': '.test.real'}

    return X_train, X_test, y_train, y_test


def read_raw_simulations(test_size=0.1, data_version=2, normalize=True, label_type='event-noevent',
                         clean_existing_data=True):
    """ Generate data files to be used by HTK

    Notes
    -----
    Data is resampeled to 10 minutes period

    """
    dataset_name = 'malte-uhma'

    train_data_path = interim_data_path / f'{dataset_name}_train'
    test_data_path = interim_data_path / f'{dataset_name}_test'
    train_D_A_data_path = interim_data_path / f'{dataset_name}_train_D_A'
    test_D_A_data_path = interim_data_path / f'{dataset_name}_test_D_A'
    train_labels_path = interim_data_path / f'{dataset_name}_labels_train'
    test_labels_path = interim_data_path / f'{dataset_name}_labels_test'

    try:
        os.mkdir(train_data_path)
        os.mkdir(test_data_path)
        os.mkdir(train_D_A_data_path)
        os.mkdir(test_D_A_data_path)
        os.mkdir(train_labels_path)
        os.mkdir(test_labels_path)
    except FileExistsError:
        if clean_existing_data:
            clean_dir(train_data_path)
            clean_dir(test_data_path)
            clean_dir(train_D_A_data_path)
            clean_dir(test_D_A_data_path)
            clean_dir(train_labels_path)
            clean_dir(test_labels_path)
        else:
            pass

    # Convert and split file into test.synth and train.synth
    for file in (raw_data_path / 'malte-uhma').glob(f'*{data_version}-*.h5'):
        file_name = file.stem[2:]

        # Read in size distribution data
        size_dist_df = pd.read_hdf(raw_data_path / 'malte-uhma' / file, key='obs/particle').resample('10T').mean() / 1e6

        if normalize:
            # size_dist_df = pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df / 1e6))
            size_dist_df = np.log10(
                np.absolute(pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df)) + 10))

        # Calculate labels
        if label_type == 'event-noevent':
            labels = get_labels_ene(file)
        elif label_type == 'nccd':
            labels = get_labels_nccd(file)

        # Split data, test/train
        if np.random.rand() < test_size:
            fo = test_data_path / file_name[2:]
            fo_label = test_labels_path / file_name[2:]
        else:
            fo = train_data_path / file_name[2:]
            fo_label = train_labels_path / file_name[2:]

        # Write data
        write_data(fo, size_dist_df)

        # Write labels to file
        write_label(fo_label, labels)

    train_file, test_file = gen_scp_files(dataset_name)

    train_label_file, test_label_file = master_label_file(dataset_name)

    logging.info('Adding deltas and acelerations...')

    test_count = HCopy(['-C', htk_misc_dir / 'config.hcopy',
                        '-S', interim_data_path / 'test_hcopy.scp',
                        '-T', 1])
    logging.info("Test files:\t" + str(test_count))

    # Train
    train_count = HCopy(['-C', htk_misc_dir / 'config.hcopy',
                         '-S', interim_data_path / 'train_hcopy.scp',
                         '-T', 1])

    logging.info("Train files:\t" + str(train_count))

    X_train = {'script_file': train_file, 'count': train_count, 'id': f'{data_version}.train.synth.synth'}
    X_test = {'script_file': test_file, 'count': test_count, 'id': f'{data_version}.test.synth.synth'}

    y_train = {'mlf': train_label_file, 'count': None, 'id': f'{data_version}.train.synth.synth'}
    y_test = {'mlf': test_label_file, 'count': None, 'id': f'{data_version}.test.synth.synth'}

    return X_train, X_test, y_train, y_test


def clean_interim():
    clean_dir(interim_data_path)
    clean_dir(figures_path)

    # os.mkdir(LABEL_TEST_PATH)
    # os.mkdir(LABEL_TRAIN_PATH)
    # os.mkdir(RESULTS_PATH)
    # os.mkdir(DATA_TEST_PATH)
    # os.mkdir(DATA_TRAIN_PATH)
    # os.mkdir(DATA_TEST_DA_PATH)
    # os.mkdir(DATA_TRAIN_DA_PATH)


def gen_scp_files(dataset_name):
    logging.info('Generating script (.scp) files...')

    train_data_path = interim_data_path / f'{dataset_name}_train'
    test_data_path = interim_data_path / f'{dataset_name}_test'
    train_D_A_data_path = interim_data_path / f'{dataset_name}_train_D_A'
    test_D_A_data_path = interim_data_path / f'{dataset_name}_test_D_A'

    train_scp_file = interim_data_path / f'{dataset_name}_train_D_A.scp'
    test_scp_file = interim_data_path / f'{dataset_name}_test_D_A.scp'

    with open(interim_data_path / 'train_hcopy.scp', 'wt') as fi:
        for file in train_data_path.glob('*'):
            line = str(train_data_path / file) + ' ' + str(train_D_A_data_path / file) + '\n'
            logging.debug(line)
            fi.write(line)

    with open(train_scp_file, 'wt') as fi:
        for file in train_data_path.glob('*'):
            line = str(train_data_path / file) + '\n'
            fi.write(line)

    with open(interim_data_path / 'test_hcopy.scp', 'wt') as fi:
        for file in test_data_path.glob('*'):
            line = str(test_data_path / file) + ' ' + str(test_data_path / file) + '\n'
            fi.write(line)

    with open(test_scp_file, 'wt') as fi:
        for file in test_data_path.glob('*'):
            line = os.path.join(test_D_A_data_path, file) + '\n'
            fi.write(line)

    return train_scp_file, test_scp_file


if __name__ == '__main__':
    SEARCH_PARAMS = {'normalize': True,
                     'data_version': '2',

                     # Labels
                     'label_type': 'event-noevent',
                     'nuc_threshold': 0.15,  # 1/cm3/10min
                     'pos_vol_threshold': 200,  # 1/m3^3/10min
                     'neg_vol_threshold': -5000,  # 1/cm3/10min

                     # Initialization
                     'variance_floor': 0.1,
                     'minimum_variance': 0.1
                     }

    hyperparameters = {'init_metho': 'HCompV',
              'raw_data_source': 'malte-uhma',
              **SEARCH_PARAMS}

    read_raw_dmps()