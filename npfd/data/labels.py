import logging
import os

import numpy as np
import pandas as pd

from npfd.models.HTK.htktools import clean_dir
from npfd.data.size_distribution import decimalDOY2datetime


def make_labels(thresholds, how='event-noevent', data_version=2):
    """Generate labels

    For each .h5 file in data/raw directory generates the corresponding HTK label file.

    Parameters
    ----------
    thresholds : int
        Description of arg1
    how : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    """
    data_version = str(data_version)

    # Remove all files from interim/labels directory
    clean_dir('../data/interim/labels')

    logging.info('Creating labels ...')

    # Walk over all files in data/raw directory.
    for file in os.listdir('../data/raw/simulation/'):
        if file.endswith('.h5') and file.startswith(data_version):
            file_name = file[:-3]

            # Calculate labels
            if how == 'event-noevent':
                labels = get_labels_ene(file, thresholds)
            elif how == 'nccd':
                labels = get_labels_nccd(file, thresholds)

            # Write labels to file
            fo = os.path.join('../data/interim/labels', file_name[2:])
            write_label(fo, labels)

    logging.info('Labels created OK!')

    results = {'mlf': master_label_file()}

    return results


def get_labels_ene(fi, thresholds):
    """Generate labels

    Generate the labels corresponding to a specific file.

    Parameters
    ----------
    fi : file
        Description of arg1
    thresholds : float
        Threshold used to detect nucleation event

    Returns
    -------
    bool
        Description of return value

    """
    # Read in size distribution data
    aux_df = pd.read_hdf('../data/raw/simulation/' + fi, key='aux')
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

    number_event = delta_N_T > thresholds['nuc_threshold']
    volume_event = (delta_V_T > thresholds['pos_vol_threshold']) | (delta_V_T < thresholds['neg_vol_threshold'])

    label_idx = np.where(number_event | volume_event, 'e', 'ne')

    labeled_df = pd.DataFrame(index=number_labels.index, data=label_idx, columns=['label'])

    # Filtro los eventos que duran una sola unidad de tiempo para evitar un pico indeseado.
    for i in range(1, labeled_df.shape[0] - 1):
        if (labeled_df.iloc[i].label != labeled_df.iloc[i - 1].label) & (
                labeled_df.iloc[i].label != labeled_df.iloc[i + 1].label):
            labeled_df.iloc[i] = labeled_df.iloc[i - 1]
    return labeled_df


def get_labels_nccd(fi, thresholds):
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
    aux_df = pd.read_hdf('../data/raw/' + fi, key='aux')
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

    label_idx = np.where(aux_df['dvol_dep'].abs() > thresholds['dep_threshold'], 'dep', 'equ')
    label_idx = np.where(aux_df['dn_coa'].abs() > thresholds['coa_threshold'], 'coa', label_idx)
    label_idx = np.where(aux_df['dvol_con'].abs() > thresholds['con_threshold'], 'con', label_idx)
    label_idx = np.where(aux_df['dn_nuc'].abs() > thresholds['nuc_threshold'], 'nuc', label_idx)

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


def master_label_file():
    logging.info("Generating Master Label File...")
    with open('../data/interim/labels.mlf', 'wt') as fo:
        # Write MLF Header
        fo.write('#!MLF!#\n')

        # Write label for each label file
        for file in os.listdir('../data/interim/labels/'):
            with open(os.path.join('../data/interim/labels', file), 'rt') as label_file:
                fo.write("\"*/" + file + ".lab\"\n")
                fo.write(label_file.read())
                fo.write('\n.\n')

    return '../data/interim/labels.mlf'


def read_dmps_maual_labels():
    dmps_labels = pd.read_csv('../data/raw/dmps/event_classification.nuk', delim_whitespace=True,
                              names=['doy', 'event_type_1', 'ddoy_start_event_1', 'ddoy_end_event_1', 'event_type_2',
                                     'ddoy_start_event_2', 'ddoy_end_event_2'])

    dmps_labels.index = decimalDOY2datetime(dmps_labels['doy'])
    dmps_labels = dmps_labels.reindex(pd.date_range('1/1/2017', '1/1/2018'))

    labels = pd.DataFrame(index=pd.date_range(start='1/1/2017', end='1/1/2018', freq='10T'), columns=['label'])

    with open('../data/raw/dmps/manual_labels.mlf', 'wt') as fo:

        # Write MLF Header
        fo.write('#!MLF!#\n')

        for i, day in dmps_labels.iterrows():
            fo.write("\"*/" + i.strftime('%Y%m%d') + ".lab\"\n")
            if not pd.isna(day.ddoy_start_event_2):
                begin = decimalDOY2datetime(day.doy)
                start_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_start_event_1) - begin).seconds)
                end_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_end_event_1) - begin).seconds)
                start_e2_in_seconds = str((decimalDOY2datetime(day.ddoy_start_event_2) - begin).seconds)
                end_e2_in_seconds = str((decimalDOY2datetime(day.ddoy_end_event_2) - begin).seconds + \
                                        (abs((decimalDOY2datetime(day.ddoy_end_event_2) - begin).days) * 86400))

                if (decimalDOY2datetime(day.ddoy_start_event_1) - begin).days < 0:
                    fo.write('0 ' + end_e1_in_seconds + ' e\n')
                else:
                    fo.write('0 ' + start_e1_in_seconds + ' ne\n')
                    fo.write(start_e1_in_seconds + ' ' + end_e1_in_seconds + ' e\n')

                fo.write(end_e1_in_seconds + ' ' + start_e2_in_seconds + ' ne\n')

                if int(end_e2_in_seconds) > 86400:
                    fo.write(start_e2_in_seconds + ' 86400 e\n')
                else:
                    fo.write(start_e2_in_seconds + ' ' + end_e2_in_seconds + ' e\n')
                    fo.write(end_e2_in_seconds + ' 86400 ne\n')

            elif not pd.isna(day.ddoy_start_event_1):
                begin = decimalDOY2datetime(day.doy)
                start_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_start_event_1) - begin).seconds)
                end_e1_in_seconds = str((decimalDOY2datetime(day.ddoy_end_event_1) - begin).seconds)

                if (decimalDOY2datetime(day.ddoy_start_event_1) - begin).days < 0:
                    fo.write('0 ' + end_e1_in_seconds + ' e\n')
                else:
                    fo.write('0 ' + start_e1_in_seconds + ' ne\n')
                    # fo.write(start_e1_in_seconds + ' ' + end_e1_in_seconds + ' e\n')

                if (decimalDOY2datetime(day.ddoy_end_event_1) - begin).days == 1:
                    fo.write(start_e1_in_seconds + ' 86400 e\n')
                else:
                    fo.write(start_e1_in_seconds + ' ' + end_e1_in_seconds + ' e\n')
                    fo.write(end_e1_in_seconds + ' 86400 ne\n')

            else:
                fo.write('0 86400 ne\n')

            fo.write('.\n')

    return {'mlf': '../data/raw/dmps/manual_labels.mlf'}


if __name__ == '__main__':
    read_dmps_maual_labels()
