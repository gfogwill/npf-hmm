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

EVENT_CLASSIFICATION_CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/dmps/event_classification.csv')
DMPS_TEST_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/dmps/dmps_mbio_2015/DATA/')
RAW_DMPS_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/dmps/inv/')

CONFIG_HCOPY_FILE_PATH = os.path.join(os.path.dirname(__file__), '../models/HTK/misc/config.hcopy')

TEST_SCP = os.path.join(os.path.dirname(__file__), '../../data/interim/test.scp')
TEST_HCOPY_SCP = os.path.join(os.path.dirname(__file__), '../../data/interim/test_hcopy.scp')
TRAIN_SCP = os.path.join(os.path.dirname(__file__), '../../data/interim/train.scp')
TRAIN_HCOPY_SCP = os.path.join(os.path.dirname(__file__), '../../data/interim/train_hcopy.scp')

DATA_TRAIN_DA_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/train_D_A')
DATA_TEST_DA_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/test_D_A')
DATA_TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/train.synth')
DATA_TEST_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/test.synth')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/results')
LABEL_TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_train')
LABEL_TEST_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_test')
LABEL_REAL_TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_real_train')
LABEL_REAL_TEST_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_real_test')
INTERIM_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/interim/')
REPORT_FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../reports/figures')

RAW_SIMULATION_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/simulation/')

ADAPT_FILES = ['20170115.cle',
               '20170204.cle',
               '20170324.cle',
               '20171117.cle',
               '20171225.cle',
               '20170220.cle',
               '20170130.cle',
               '20171202.cle',
               ]


def make_dataset(hyperparameters, clean_interim_dir = False, test_size=0.1):
    r"""Generates data files

    Long description

    Parameters
    ----------
    which : {'simulation', 'real'}, optional
        Choices in brackets, default first when optional.
    *args : iterable
        Other arguments.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    X :
    y :

    Examples
    --------

    """

    # Remove all files from interim directory
    if clean_interim_dir:
        clean_interim()

    if hyperparameters['raw_data_source'] == 'simulation':
        logging.info('Converting simulation raw files to HTK format ...')
        X_train, X_test, y_train, y_test = read_raw_simulations(hyperparameters, test_size)

    elif hyperparameters['raw_data_source'] == 'real':
        logging.info('Converting real raw files to HTK format ...')
        X_train, X_test, y_train, y_test = read_raw_dmps(test_size=test_size)

    else:
        raise ValueError("Invalid option: " + hyperparameters['raw_data_source'])

    return X_train, X_test, y_train, y_test


def read_raw_dmps(skip_invalid_day=False, clean_existing_data=True, test_size=0.1):
    train_data_path = os.path.join(INTERIM_DATA_DIR, 'train.synth.real')
    test_data_path = os.path.join(INTERIM_DATA_DIR, 'test.synth.real')
    train_D_A_data_path = os.path.join(INTERIM_DATA_DIR, 'train_D_A.real')
    test_D_A_data_path = os.path.join(INTERIM_DATA_DIR, 'test_D_A.real')

    train_scp_file = os.path.join(INTERIM_DATA_DIR, 'train_D_A.real.scp')
    test_scp_file = os.path.join(INTERIM_DATA_DIR, 'test_D_A.real.scp')

    try:
        os.mkdir(train_data_path)
        os.mkdir(test_data_path)
        os.mkdir(train_D_A_data_path)
        os.mkdir(test_D_A_data_path)
    except FileExistsError:
        # pass
        if clean_existing_data:
            clean_dir(train_data_path)
            clean_dir(test_data_path)
            clean_dir(train_D_A_data_path)
            clean_dir(test_D_A_data_path)
        else:
            pass

    labels = pd.read_csv(EVENT_CLASSIFICATION_CSV_PATH, index_col=0)
    labels.index = pd.to_datetime(labels.index)
    count = 0
    for file in os.listdir(RAW_DMPS_PATH):
        if file.endswith('.cle'):

            nukdata = pd.read_csv(RAW_DMPS_PATH + file, sep=r'\s+')
            nukdata = nukdata.replace(np.nan, -999)
            nukdata.index = nukdata.iloc[:, 0].apply(decimalDOY2datetime)

            # TODO: change ffill
            # nukdata = nukdata.drop(columns=nukdata.columns[[0, 1]]).resample('10T').ffill()
            nukdata = nukdata.drop(columns=nukdata.columns[[0, 1]]).resample('10T').mean()
            nukdata = nukdata.replace(np.nan, -999)

            if skip_invalid_day and nukdata.isin([-999]).any().any():
                continue

            file = file.replace('DM', '')
            file = file.replace('dm', '')

            # Split data, 90% for training, 10% for testing
            # if np.random.rand() < test_size:
            if file not in ADAPT_FILES:
                fo = os.path.join(test_data_path, file[:-4])
                fo_label = os.path.join(LABEL_REAL_TEST_PATH, file[:-4])
                fo_D_A = fo.replace('test.synth', 'test_D_A')
            else:
                fo = os.path.join(train_data_path, file[:-4])
                fo_label = os.path.join(LABEL_REAL_TRAIN_PATH, file[:-4])
                fo_D_A = fo.replace('train.synth', 'train_D_A')

            # labels.index = pd.to_datetime(labels.index)
            # # Get labels
            # start = datetime.datetime(int(file[:4]), int(file[4:6]), int(file[6:8]))
            # end = (datetime.datetime(int(file[:4]), int(file[4:6]), int(file[6:8])) + datetime.timedelta(days=1))
            # day_labels = labels.loc[start:end]

            day_labels = labels.loc[nukdata.index]

            day_labels.where(nukdata.min(axis=1) != -999, 'na', inplace=True)

            # # Write labels to file
            write_label(fo_label, day_labels)

            write_data(fo, np.log10(np.absolute(nukdata + 10)))
            HCopy([fo, fo_D_A, '-C', CONFIG_HCOPY_FILE_PATH])

            # scp_file.write(fo_D_A + '\n')
            count += 1

    train_labels, test_labels = dmps_master_label_file()

    with open(train_scp_file, 'wt') as fi:
        for file in os.listdir(os.path.join(INTERIM_DATA_DIR, 'train_D_A.real')):
            line = os.path.join(os.path.join(INTERIM_DATA_DIR, 'train_D_A.real'), file) + '\n'
            fi.write(line)

    with open(test_scp_file, 'wt') as fi:
        for file in os.listdir(os.path.join(INTERIM_DATA_DIR, 'test_D_A.real')):
            line = os.path.join(os.path.join(INTERIM_DATA_DIR, 'test_D_A.real'), file) + '\n'
            fi.write(line)

    X_train = {'script_file': train_scp_file, 'count': 1, 'id': 'train_D_A.real'}
    X_test = {'script_file': test_scp_file, 'count': 1, 'id': 'test_D_A.real'}

    y_train = {'mlf': train_labels, 'count': None, 'id': 'train.real'}
    y_test = {'mlf': test_labels, 'count': None, 'id': '.test.real'}

    return X_train, X_test, y_train, y_test


def read_raw_simulations(params, test_size=0.1):
    """ Generate data files to be used by HTK

    Notes
    -----
    Data is resampeled to 10 minutes period

    """

    if not os.listdir(DATA_TEST_PATH):
        # Convert and split file into test.synth and train.synth
        for file in os.listdir(RAW_SIMULATION_DATA_PATH):
            if file.endswith('.h5') and file.startswith(params['data_version']):
                file_name = file[:-3]

                # Read in size distribution data
                size_dist_df = pd.read_hdf(RAW_SIMULATION_DATA_PATH + file, key='obs/particle').resample('10T').mean() / 1e6

                if params['normalize']:
                    # size_dist_df = pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df / 1e6))
                    size_dist_df = np.log10(
                        np.absolute(pd.DataFrame(index=size_dist_df.index, data=cm3_to_dndlogdp(size_dist_df)) + 10))
                else:
                    pass

                # Calculate labels
                if params['label_type'] == 'event-noevent':
                    labels = get_labels_ene(file, params)
                elif params['label_type'] == 'nccd':
                    labels = get_labels_nccd(file, params)

                # Split data, 90% for training, 10% for testing
                if np.random.rand() < test_size:
                    fo = os.path.join(INTERIM_DATA_DIR, 'test.synth', file_name[2:])
                    fo_label = os.path.join(LABEL_TEST_PATH, file_name[2:])
                else:
                    fo = os.path.join(INTERIM_DATA_DIR, 'train.synth', file_name[2:])
                    fo_label = os.path.join(LABEL_TRAIN_PATH, file_name[2:])

                # Write data
                write_data(fo, size_dist_df)

                # Write labels to file
                write_label(fo_label, labels)

    train_file, test_file = gen_scp_files()
    train_label_file, test_label_file = master_label_file()

    logging.info('Adding deltas and acelerations...')

    test_count = HCopy(['-C', CONFIG_HCOPY_FILE_PATH,
                        '-S', TEST_HCOPY_SCP,
                        '-T', 1])
    logging.info("Test files:\t" + str(test_count))

    # Train
    train_count = HCopy(['-C', CONFIG_HCOPY_FILE_PATH,
                         '-S', TRAIN_HCOPY_SCP,
                         '-T', 1])
    logging.info("Train files:\t" + str(train_count))

    X_train = {'script_file': train_file, 'count': train_count, 'id': params['data_version'] + '.train.synth.synth'}
    X_test = {'script_file': test_file, 'count': test_count, 'id': params['data_version'] + '.test.synth.synth'}

    y_train = {'mlf': train_label_file, 'count': None, 'id': params['data_version'] + '.train.synth.synth'}
    y_test = {'mlf': test_label_file, 'count': None, 'id': params['data_version'] + '.test.synth.synth'}

    return X_train, X_test, y_train, y_test


def clean_interim():
    clean_dir(INTERIM_DATA_DIR)
    clean_dir(REPORT_FIGURES_DIR)

    os.mkdir(LABEL_TEST_PATH)
    os.mkdir(LABEL_TRAIN_PATH)
    os.mkdir(RESULTS_PATH)
    os.mkdir(DATA_TEST_PATH)
    os.mkdir(DATA_TRAIN_PATH)
    os.mkdir(DATA_TEST_DA_PATH)
    os.mkdir(DATA_TRAIN_DA_PATH)


def gen_scp_files():
    logging.info('Generating script (.scp) files...')

    with open(TRAIN_HCOPY_SCP, 'wt') as fi:
        for file in os.listdir(DATA_TRAIN_PATH):
            line = os.path.join(DATA_TRAIN_PATH, file) + ' ' + os.path.join(DATA_TRAIN_DA_PATH, file) + '\n'
            logging.debug(line)
            fi.write(line)

    with open(TRAIN_SCP, 'wt') as fi:
        for file in os.listdir(DATA_TRAIN_PATH):
            line = os.path.join(DATA_TRAIN_DA_PATH, file) + '\n'
            fi.write(line)

    with open(TEST_HCOPY_SCP, 'wt') as fi:
        for file in os.listdir(DATA_TEST_PATH):
            line = os.path.join(DATA_TEST_PATH, file) + ' ' + os.path.join(DATA_TEST_DA_PATH, file) + '\n'
            fi.write(line)

    with open(TEST_SCP, 'wt') as fi:
        for file in os.listdir(DATA_TEST_PATH):
            line = os.path.join(DATA_TEST_DA_PATH, file) + '\n'
            fi.write(line)

    return TRAIN_SCP, TEST_SCP




# def read_raw_dmps_old(skip_invalid_day=True):
#     train_id = '2017.train.synth.real'
#
#     scp_file_name = INTERIM_DATA_DIR + train_id.replace('train.synth', 'train_D_A') + '.scp'
#
#     try:
#         os.mkdir(os.path.join(INTERIM_DATA_DIR, train_id.replace('train.synth', 'train_D_A')))
#         os.mkdir(os.path.join(INTERIM_DATA_DIR, train_id))
#     except FileExistsError:
#         # pass
#         with open(scp_file_name, 'wt') as scp_file:
#             count = 0
#             for file in os.listdir(DMPS_TRAIN_PATH):
#
#                 if file.endswith('.cle'):
#                     nukdata = pd.read_csv(DMPS_TRAIN_PATH + file, sep=r'\s+')
#                     nukdata = nukdata.replace(np.nan, -999)
#                     nukdata.index = nukdata.iloc[:, 0].apply(decimalDOY2datetime)
#
#                     # TODO: change ffill
#                     # nukdata = nukdata.drop(columns=nukdata.columns[[0, 1]]).resample('10T').ffill()
#                     nukdata = nukdata.drop(columns=nukdata.columns[[0, 1]]).resample('10T').mean()
#                     nukdata = nukdata.replace(np.nan, -999)
#
#                     if skip_invalid_day or nukdata.isin([-999]).any().any():
#                         continue
#
#                     file = file.replace('DM', '')
#                     file = file.replace('dm', '')
#
#                     fo = os.path.join(INTERIM_DATA_DIR, train_id, file[:-4])
#                     fo_D_A = fo.replace('train.synth', 'train_D_A')
#
#                     write_data(fo, np.log10(np.absolute(nukdata + 10)))
#                     HCopy([fo, fo_D_A, '-C', CONFIG_HCOPY_FILE_PATH])
#
#                     scp_file.write(fo_D_A + '\n')
#                     count += 1
#
#     X_train = {'script_file': scp_file_name, 'count': 1, 'id': train_id.replace('train.synth', 'train_D_A')}
#
#     test_id = '2015.test.synth.real'
#     try:
#         os.mkdir(os.path.join(INTERIM_DATA_DIR, test_id.replace('test.synth', 'test_D_A')))
#         os.mkdir(os.path.join(INTERIM_DATA_DIR, test_id))
#     except FileExistsError:
#         pass
#
#     scp_file_name = INTERIM_DATA_DIR + test_id.replace('test.synth', 'test_D_A') + '.scp'
#     with open(scp_file_name, 'wt') as scp_file:
#         count = 0
#         for file in os.listdir(DMPS_TEST_PATH):
#             if file.endswith('.cle'):
#                 nukdata = pd.read_csv(DMPS_TEST_PATH + file, sep=r'\s+')
#                 nukdata = nukdata.replace(np.nan, -999)
#                 nukdata.index = nukdata.iloc[:, 0].apply(decimalDOY2datetime)
#                 nukdata = nukdata.drop(columns=nukdata.columns[[0, 1]]).resample('10T').mean()
#                 nukdata = nukdata.replace(np.nan, -999)
#
#                 if nukdata.isin([-999]).any().any():
#                     continue
#
#                 file = file.replace('DM', '')
#                 file = file.replace('dm', '')
#                 fo = os.path.join(INTERIM_DATA_DIR, test_id, file[:-4])
#                 fo_D_A = fo.replace('test.synth', 'test_D_A')
#                 write_data(fo, np.log10(np.absolute(nukdata + 10)))
#                 HCopy([fo, fo_D_A, '-C', CONFIG_HCOPY_FILE_PATH])
#
#                 scp_file.write(fo_D_A + '\n')
#                 count += 1
#     X_test = {'script_file': scp_file_name, 'count': count, 'id': test_id.replace('test.synth', 'test_D_A')}
#
#     return X_train, X_test

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
              'raw_data_source': 'simulation',
              **SEARCH_PARAMS}

    read_raw_dmps()