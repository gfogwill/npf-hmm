import logging
import os

import numpy as np
import pandas as pd

from npfd.models.HTK.htktools import clean_dir
from npfd.data.size_distribution import decimalDOY2datetime

from ..paths import raw_data_path, interim_data_path

MANUAL_LABELS_MLF_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/manual_labels_real_data.mlf')

TEST_LABELS_MLF = os.path.join(os.path.dirname(__file__), '../../data/interim/test_labels.mlf')
TRAIN_LABELS_MLF = os.path.join(os.path.dirname(__file__), '../../data/interim/train_labels.mlf')

DMPS_TEST_LABELS_MLF = os.path.join(os.path.dirname(__file__), '../../data/interim/dmps_test_labels.mlf')
DMPS_TRAIN_LABELS_MLF = os.path.join(os.path.dirname(__file__), '../../data/interim/dmps_train_labels.mlf')
DMPS_LABEL_TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_real_train')
DMPS_LABEL_TEST_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_real_test')

RAW_SIMULATION_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/malte-uhma/')
LABELS_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels')

LABEL_TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_train')
LABEL_TEST_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/labels_test')


# def make_labels(thresholds, how='event-noevent', data_version=2):
#     """Generate labels
#
#     For each .h5 file in data/raw directory generates the corresponding HTK label file.
#
#     Parameters
#     ----------
#     thresholds : int
#         Description of arg1
#     how : str
#         Description of arg2
#
#     Returns
#     -------
#     bool
#         Description of return value
#
#     """
#     # Remove all files from interim/labels directory
#     clean_dir(LABELS_PATH)
#
#     logging.info('Creating labels ...')
#
#     # for file in os.listdir(RAW_SIMULATION_DATA_PATH):
#     #     if file.endswith('.h5') and file.startswith(data_version):
#     # Walk over all files in data/raw directory.
#     for file in (raw_data_path / 'malte-uhma').glob(f'*{data_version}-*.h5'):
#         file_name = file[:-3]
#
#         # Calculate labels
#         if how == 'event-noevent':
#             labels = get_labels_ene(file, thresholds)
#         elif how == 'nccd':
#             labels = get_labels_nccd(file, thresholds)
#
#         # Write labels to file
#         fo = os.path.join(LABELS_PATH, file_name[2:])
#         write_label(fo, labels)
#
#     logging.info('Labels created OK!')
#
#     results = {'script_file': master_label_file()}
#
#     return results


def get_labels_ene(fi, nuc_threshold=0.15, pos_vol_threshold=200, neg_vol_threshold=-5000):
    """Generate event/no-event labels

    Generate the labels corresponding to a specific file.

    Parameters
    ----------
    fi : file
        Description of arg1
    nuc_threshold : float
        Threshold used to detect nucleation event
    pos_vol_threshold : float
        Threshold used to detect nucleation event
    neg_vol_threshold : float
        Threshold used to detect nucleation event
    Returns
    -------
    bool
        DataFrame with labels

    """
    # Read in size distribution data
    aux_df = pd.read_hdf(RAW_SIMULATION_DATA_PATH + fi, key='aux')
    aux_df = aux_df.resample('10T').sum()

    aux_df[['dn_nuc', 'dn_coa', 'dn_dep']] = aux_df[['dn_nuc', 'dn_coa', 'dn_dep']] / 1e6
    number_labels = aux_df[['dn_nuc', 'dn_coa', 'dn_dep']].abs().idxmax(axis=1)
    number_labels = number_labels.replace(['dn_nuc', 'dn_coa', 'dn_dep'], ['nuc', 'coa', 'dep'])

    volume_labels = aux_df[['dvol_nuc', 'dvol_con', 'dvol_dep']].abs().idxmax(axis=1)
    volume_labels = volume_labels.replace(['dvol_nuc', 'dvol_con', 'dvol_dep'], ['nuc', 'con', 'dep'])

    delta_N_T = aux_df[['dn_coa', 'dn_dep']].abs().sum(axis=1)
    delta_V_T = aux_df[['dvol_dep']].sum(axis=1)

    delta_N_T = delta_N_T.where(delta_N_T.values < 100, 0)
    # delta_V_T = delta_V_T.where(delta_V_T.values < 1000, 0)

    number_event = delta_N_T > nuc_threshold
    volume_event = (delta_V_T > pos_vol_threshold) | (
            delta_V_T < neg_vol_threshold)

    label_idx = np.where(number_event | volume_event, 'e', 'ne')

    labeled_df = pd.DataFrame(index=number_labels.index, data=label_idx, columns=['label'])

    # Filtro los eventos que duran una sola unidad de tiempo para evitar un pico indeseado.
    for i in range(1, labeled_df.shape[0] - 1):
        if (labeled_df.iloc[i].label != labeled_df.iloc[i - 1].label) & (
                labeled_df.iloc[i].label != labeled_df.iloc[i + 1].label):
            labeled_df.iloc[i] = labeled_df.iloc[i - 1]
    return labeled_df


def get_labels_nccd(fi, hyperparameters):
    """Generate labels

    Generate the labels corresponding to a specific file.

    Parameters
    ----------
    fi : file
        Description of arg1
    nuc_threshold : float
        Threshold used to detect nucleation event
    vol_threshold : float
        Threshold used to detect volume event

    Returns
    -------
    bool
        Description of return value

    """
    # Read in size distribution data
    aux_df = pd.read_hdf(raw_data_path / fi, key='aux')
    aux_df = aux_df.resample('10T').sum()

    aux_df[['dn_nuc', 'dn_coa', 'dn_dep']] = aux_df[['dn_nuc', 'dn_coa', 'dn_dep']] / 1e6
    number_labels = aux_df[['dn_nuc', 'dn_coa', 'dn_dep']].abs().idxmax(axis=1)
    number_labels = number_labels.replace(['dn_nuc', 'dn_coa', 'dn_dep'], ['nuc', 'coa', 'dep'])

    volume_labels = aux_df[['dvol_nuc', 'dvol_con', 'dvol_dep']].abs().idxmax(axis=1)
    volume_labels = volume_labels.replace(['dvol_nuc', 'dvol_con', 'dvol_dep'], ['nuc', 'con', 'dep'])

    delta_N_T = aux_df[['dn_nuc', 'dn_coa', 'dn_dep']].abs().sum(axis=1)
    delta_V_T = aux_df[['dvol_nuc', 'dvol_con', 'dvol_dep']].sum(axis=1)

    # number_event = (delta_N_T / 1e6) > nuc_threshold
    # volume_event = (delta_V_T > pos_vol_threshold) | (delta_V_T < neg_vol_threshold)

    # label_idx = np.where(volume_event, volume_labels, 'equ')
    # label_idx = np.where(number_event, number_labels, label_idx)

    label_idx = np.where(aux_df['dvol_dep'].abs() > hyperparameters['dep_threshold'], 'dep', 'equ')
    label_idx = np.where(aux_df['dn_coa'].abs() > hyperparameters['coa_threshold'], 'coa', label_idx)
    label_idx = np.where(aux_df['dvol_con'].abs() > hyperparameters['con_threshold'], 'con', label_idx)
    label_idx = np.where(aux_df['dn_nuc'].abs() > hyperparameters['nuc_threshold'], 'nuc', label_idx)

    labeled_df = pd.DataFrame(index=number_labels.index, data=label_idx, columns=['label'])

    return labeled_df


def write_label(fo_name, labels):
    labels.index = (labels.index.to_series() - labels.index.to_series().iloc[0]).values

    fo = open(fo_name, 'wt')
    begin_event = '0'
    start = labels.index[0]

    for i, g in labels.groupby([(labels.label != labels.label.shift()).cumsum()]):
        end = g.drop_duplicates(keep='last').index
        e_time = (end - start).total_seconds()
        label = g.drop_duplicates(keep='last').values
        end_event = " {:.0f} ".format(e_time[0])

        if e_time[0] == 0:
            continue

        fo.write(begin_event + end_event + label[0][0])
        begin_event = end_event[1:-1]
        if begin_event != '86400':
            fo.write('\n')
    return


def master_label_file(dataset_name=None):
    if dataset_name is None:
        raise Exception('dataset_name is required')

    train_mlf = interim_data_path / f'{dataset_name}' / '_train_labels.mlf'
    test_mlf = interim_data_path / f'{dataset_name}' / '_test_labels.mlf'
    train_labels_path = interim_data_path / f'{dataset_name}_labels_train'
    test_labels_path = interim_data_path / f'{dataset_name}_labels_test'

    logging.info("Generating Master Label File (Train)...")
    with open(train_mlf, 'wt') as fo:
        # Write MLF Header
        fo.write('#!MLF!#\n')

        # Write label for each label file
        for file in os.listdir(train_labels_path):
            with open(os.path.join(train_labels_path, file), 'rt') as label_file:
                fo.write("\"*/" + file + ".lab\"\n")
                fo.write(label_file.read())
                fo.write('\n.\n')

    logging.info("Generating Master Label File (Test)...")
    with open(test_mlf, 'wt') as fo:
        # Write MLF Header
        fo.write('#!MLF!#\n')

        # Write label for each label file
        for file in os.listdir(test_labels_path):
            with open(os.path.join(test_labels_path, file), 'rt') as label_file:
                fo.write("\"*/" + file + ".lab\"\n")
                fo.write(label_file.read())
                fo.write('\n.\n')

    return train_mlf, test_mlf


def dmps_master_label_file():

    logging.info("Generating Master Label File (Train)...")
    with open(DMPS_TRAIN_LABELS_MLF, 'wt') as fo:
        # Write MLF Header
        fo.write('#!MLF!#\n')

        # Write label for each label file
        for file in os.listdir(DMPS_LABEL_TRAIN_PATH):
            with open(os.path.join(DMPS_LABEL_TRAIN_PATH, file), 'rt') as label_file:
                fo.write("\"*/" + file + ".lab\"\n")
                fo.write(label_file.read())
                fo.write('\n.\n')

    logging.info("Generating Master Label File (Test)...")
    with open(DMPS_TEST_LABELS_MLF, 'wt') as fo:
        # Write MLF Header
        fo.write('#!MLF!#\n')

        # Write label for each label file
        for file in os.listdir(DMPS_LABEL_TEST_PATH):
            with open(os.path.join(DMPS_LABEL_TEST_PATH, file), 'rt') as label_file:
                fo.write("\"*/" + file + ".lab\"\n")
                fo.write(label_file.read())
                fo.write('\n.\n')

    return DMPS_TRAIN_LABELS_MLF, DMPS_TEST_LABELS_MLF


# def read_dmps_maual_labels():
#     dmps_labels = pd.read_csv(EVENT_CLASSIFICATION_CSV_PATH, delim_whitespace=True,
#                               names=['doy', 'event_type_1', 'ddoy_start_event_1', 'ddoy_end_event_1', 'event_type_2',
#                                      'ddoy_start_event_2', 'ddoy_end_event_2'])
#
#     dmps_labels.index = decimalDOY2datetime(dmps_labels['doy'])
#     dmps_labels = dmps_labels.reindex(pd.date_range('1/1/2017', '1/1/2018'))
#
#     labels = pd.DataFrame(index=pd.date_range(start='1/1/2017', end='1/1/2018', freq='10T'), columns=['label'])
#
#     with open(MANUAL_LABELS_MLF_PATH, 'wt') as fo:
#
#         # Write MLF Header
#         fo.write('#!MLF!#\n')
#
#         for i, day in dmps_labels.iterrows():
#             if day.name.strftime("%Y%m%d") == '20170601':
#                 print(day)
#             starts_prev_day = False
#             continue_next_day = False
#             if day.ddoy_start_event_1 < day.doy:
#                 day.ddoy_start_event_1 = day.doy
#                 starts_prev_day = True
#             if day.ddoy_end_event_1 > day.doy + 1:
#                 day.ddoy_end_event_1 = day.doy + 1
#                 continue_next_day = True
#             if day.ddoy_end_event_2 > day.doy + 1:
#                 day.ddoy_end_event_2 = day.doy + 1
#                 continue_next_day = True
#
#             fo.write("\"*/" + i.strftime('%Y%m%d') + ".lab\"\n")
#             # fo.write("\"*/" + i.strftime('%Y%m%d') + "\"\n")
#             if not pd.isna(day.ddoy_start_event_2):
#                 begin = decimalDOY2datetime(day.doy)
#                 start_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_start_event_1) - begin).seconds)
#                 end_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_end_event_1) - begin).seconds)
#                 start_e2_in_seconds = str((decimalDOY2datetime(day.ddoy_start_event_2) - begin).seconds)
#                 end_e2_in_seconds = str((decimalDOY2datetime(day.ddoy_end_event_2) - begin).seconds + \
#                                         (abs((decimalDOY2datetime(day.ddoy_end_event_2) - begin).days) * 86400))
#
#                 #if (decimalDOY2datetime(day.ddoy_start_event_1) - begin).days < 0:
#                 if starts_prev_day:
#                     fo.write('0 ' + end_e1_in_seconds + ' e\n')
#                 else:
#                     fo.write('0 ' + start_e1_in_seconds + ' ne\n')
#                     fo.write(start_e1_in_seconds + ' ' + end_e1_in_seconds + ' e\n')
#
#                 fo.write(end_e1_in_seconds + ' ' + start_e2_in_seconds + ' ne\n')
#
#                 if continue_next_day:
#                     fo.write(start_e2_in_seconds + ' 86400 e\n')
#                 else:
#                     fo.write(start_e2_in_seconds + ' ' + end_e2_in_seconds + ' e\n')
#                     fo.write(end_e2_in_seconds + ' 86400 ne\n')
#
#             elif not pd.isna(day.ddoy_start_event_1):
#                 begin = decimalDOY2datetime(day.doy)
#                 start_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_start_event_1) - begin).seconds)
#                 end_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_end_event_1) - begin).seconds)
#
#                 if starts_prev_day:
#                     fo.write('0 ' + end_e1_in_seconds + ' e\n')
#                 else:
#                     fo.write('0 ' + start_e1_in_seconds + ' ne\n')
#                     # fo.write(start_e1_in_seconds + ' ' + end_e1_in_seconds + ' e\n')
#
#                 if continue_next_day:
#                     fo.write(start_e1_in_seconds + ' 86400 e\n')
#                 else:
#                     if not starts_prev_day:
#                         fo.write(start_e1_in_seconds + ' ' + end_e1_in_seconds + ' e\n')
#                     fo.write(end_e1_in_seconds + ' 86400 ne\n')
#
#             else:
#                 fo.write('0 86400 ne\n')
#
#             fo.write('.\n')
#
#     return {'mlf': MANUAL_LABELS_MLF_PATH}, {
#         'mlf': MANUAL_LABELS_MLF_PATH}


if __name__ == '__main__':
    read_dmps_maual_labels()
