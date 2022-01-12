# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import datetime

from src.models.HTK.htktools import HCopy, clean_dir
from src.data.size_distribution import cm3_to_dndlogdp, decimalDOY2datetime
from src.data.htk import write_data
from src.data.labels import get_labels_ene, get_labels_nccd, write_label, master_label_file, dmps_master_label_file
from src.paths import raw_data_path, interim_data_path, figures_path, htk_misc_dir


def make_dataset(dataset_name=None, clean_interim_dir=True, adapt_list=None, **kwargs):
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
    if not clean_interim_dir:
        pass
    # Remove all files from interim directory
    else:
        clean_dir(interim_data_path)

    if dataset_name == 'db1' or dataset_name == 'db2':
        logging.info('Converting malte-uhma raw files to HTK format ...')
        X_train, X_test, y_train, y_test = read_raw_simulations(dataset_name, **kwargs)

    elif dataset_name == 'dmps' or dataset_name == 'dbtest':
        logging.info('Converting real raw files to HTK format ...')
        X_train, X_test, y_train, y_test = read_raw_dmps(dataset_name, adapt_list=adapt_list, **kwargs)

    else:
        raise ValueError("Invalid dataset name: " + dataset_name)

    return X_train, X_test, y_train, y_test


def read_raw_dmps(dataset_name=None, skip_invalid_day=False,
                  clean_existing_data=True, test_size=0.1, adapt_list=None, seed=None, **kwargs):

    if seed is not None:
        logging.debug(f"Using seed: {seed}")
        np.random.seed(seed)

    if dataset_name is None:
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

    labels = pd.read_csv(raw_data_path / dataset_name / 'events.csv', index_col=0)
    labels.index = pd.to_datetime(labels.index)
    train_count = 0
    test_count = 0

    for file in (raw_data_path / dataset_name).glob('./*/*.csv'):
        nukdata = pd.read_csv(file, index_col='datetime', parse_dates=['datetime'])  # .resample('10T').mean()

        if skip_invalid_day and nukdata.isin([-999]).any().any():
            continue

        if adapt_list is None:
            # Split data, 90% for training, 10% for testing
            if np.random.rand() < test_size:
                test_count += 1
                fo = test_data_path / file.stem[2:]
                fo_label = test_labels_path / file.stem[2:]
                fo_D_A = test_D_A_data_path / file.stem[2:]
            else:
                train_count += 1
                fo = train_data_path / file.stem[2:]
                fo_label = train_labels_path / file.stem[2:]
                fo_D_A = train_D_A_data_path / file.stem[2:]
        else:
            if file.stem[2:] in adapt_list:
                train_count += 1
                fo = train_data_path / file.stem[2:]
                fo_label = train_labels_path / file.stem[2:]
                fo_D_A = train_D_A_data_path / file.stem[2:]
            else:
                test_count += 1
                fo = test_data_path / file.stem[2:]
                fo_label = test_labels_path / file.stem[2:]
                fo_D_A = test_D_A_data_path / file.stem[2:]

        day_labels = labels.loc[nukdata.index[0]:nukdata.index[0]+datetime.timedelta(days=1)].copy()

        day_labels.where(nukdata.sum(axis=1) != 0, np.nan, inplace=True)

        nukdata.dropna(inplace=True)
        day_labels.dropna(inplace=True)

        day_labels.where(nukdata.min(axis=1) != -999, 'na', inplace=True)

        # # Write labels to file
        write_label(fo_label, day_labels)

        write_data(fo, np.log10(np.absolute(nukdata + 10)))

        logging.debug("Train files:\t" + str(train_count))
        logging.debug("Test files:\t" + str(test_count))

        HCopy([fo, fo_D_A, '-C', htk_misc_dir / 'config.hcopy'])

    train_labels, test_labels = master_label_file(dataset_name)

    with open(train_scp_file, 'wt') as fi:
        for file in train_D_A_data_path.glob('*'):
            line = str(train_D_A_data_path / file) + '\n'
            fi.write(line)

    with open(test_scp_file, 'wt') as fi:
        for file in test_D_A_data_path.glob('*'):
            line = str(test_D_A_data_path / file) + '\n'
            fi.write(line)

    X_train = {'script_file': train_scp_file, 'count': train_count, 'id': 'train_D_A.real'}
    X_test = {'script_file': test_scp_file, 'count': test_count, 'id': 'test_D_A.real'}

    y_train = {'mlf': train_labels, 'count': None, 'id': 'train.real'}
    y_test = {'mlf': test_labels, 'count': None, 'id': 'test.real'}

    return X_train, X_test, y_train, y_test


def read_raw_simulations(dataset_name=None, test_size=0.1, data_version=2, normalize=True, label_type='event-noevent',
                         clean_existing_data=True, add_na=True, **kwargs):
    """ Generate data files to be used by HTK

    Notes
    -----
    Data is resampeled to 10 minutes period

    """

    # np.random.seed(7)

    if dataset_name is None:
        raise Exception('dataset_name is needed')

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
    for file in (raw_data_path / f'{dataset_name}').glob(f'*{data_version}-*.h5'):
        file_name = file.stem[2:]

        # Read in size distribution data
        size_dist_df = pd.read_hdf(raw_data_path / f'{dataset_name}' / file, key='obs/particle').resample('10T').mean() / 1e6

        if normalize:
            # size_dist_df = pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df / 1e6))
            size_dist_df = np.log10(
                np.absolute(pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df)) + 10))

        # Calculate labels
        if label_type == 'event-noevent':
            labels = get_labels_ene(file, add_na)
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
    import src.visualization as viz

    X_train, X_val, y_train, y_val = make_dataset(dataset_name='dmps', clean_interim_dir=True, test_size=1)
    viz.visualize.generate_plots('real_data', X_val, y_val)