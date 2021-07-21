# -*- coding: utf-8 -*-
import os
import pathlib

import pandas as pd
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from npfd.data.size_distribution import cm3_to_dndlogdp
from npfd.data.dataset import clean_dir
from npfd.data.htk import read_data
from npfd.data.labels import get_labels_ene

import shutil

from ..paths import figures_path, interim_data_path


DATA_TEST_DA_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/test_D_A')
RESULTS_MLF_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/results.mlf')


def generate_plots(out_dir, X, y1=None, y2=None):

    if os.path.exists(figures_path / out_dir):
        shutil.rmtree(figures_path / out_dir)

    os.mkdir(figures_path / out_dir)

    with open(X['script_file'], 'rt') as files_list:

        if y2 is not None:
            for file in files_list.read().splitlines():
                plot_X_y1_y2(pathlib.Path(file), out_dir, y1, y2)

        elif y1 is not None:
            for file in files_list.read().splitlines():
                plot_X_y1(pathlib.Path(file), out_dir, y1)

        else:
            for file in files_list.read().splitlines():
                plot_X(pathlib.Path(file), out_dir)

    return


def plot_X(file, out_dir=None):
    _, obs, delta, acc = read_data(file)
    f = plt.figure()
    plt.pcolor(obs.values[::1, ::1].T, cmap='jet')
    plt.colorbar()
    plt.clim(0, 4)
    plt.title(pd.to_datetime(file.stem).strftime('%Y-%m-%d'))

    if out_dir is not None:
        plt.savefig(figures_path / out_dir / file.stem)
        f.clear()
        plt.close(f)
    else:
        plt.show()


def plot_X_y1(file, out_dir, y1):
    _, obs, delta, acc = read_data(file)
    file_length = obs.__len__()
    label_start, label_end, labels = read_mlf_label(y1['mlf'], file.stem, file_length)
    label_start.head()
    label_end.head()
    f = plt.figure()
    size = f.get_size_inches()
    lw = size[0]*72/file_length
    ax1 = plt.subplot2grid((12, 12), (0, 0), rowspan=10, colspan=12)
    plt.pcolor(obs.values[::1, ::1].T, cmap='jet')
    plt.clim(0, 4)
    plt.title(pd.to_datetime(file.stem).strftime('%Y-%m-%d'))
    ax1.axes.get_xaxis().set_visible(False)
    ax2 = plt.subplot2grid((12, 12), (10, 0), rowspan=2, colspan=12)
    ax2.plot(label_start.index, np.ones(label_start.index.shape[0]))
    for t, label in zip(label_end.index, label_end.values):
        if label == 'equ' or label == 'ne':
            ax2.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax2.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax2.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax2.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax2.axvline(t, color='y', linewidth=lw)
        if label == 'e':
            ax2.axvline(t, color='c', linewidth=lw)
        if label == 'na':
            ax2.axvline(t, color='w', linewidth=lw)
    ax2.axes.get_yaxis().set_visible(False)
    # ax2.axes.get_xaxis().set_visible(False)
    plt.xlim([label_start.index[0], label_end.index[-1]])
    plt.savefig(figures_path / out_dir / file.stem)
    # plt.show()
    f.clear()
    plt.close(f)


def plot_X_y1_y2(file, out_dir, y1, y2, show=False):
    _, obs, delta, acc = read_data(file)
    file_length = obs.__len__()
    label1_start, label1_end, labels1 = read_mlf_label(y1['mlf'], file.stem, file_length)
    label2_start, label2_end, labels2 = read_mlf_label(y2['mlf'], file.stem, file_length)
    # label1_start, label1_end, labels1 = read_mlf_label(y1['mlf'], file.stem, file_length)
    # label2_start, label2_end, labels2 = read_mlf_label(y2['mlf'], file.stem, file_length)
    f = plt.figure()
    size = f.get_size_inches()
    lw = size[0]*72/file_length
    ax1 = plt.subplot2grid((14, 12), (0, 0), rowspan=10, colspan=12)
    plt.pcolor(obs.values[::1, ::1].T, cmap='jet')
    plt.clim(0, 4)
    plt.title(pd.to_datetime(file.stem).strftime('%Y-%m-%d'))
    ax1.axes.get_xaxis().set_visible(False)
    ax2 = plt.subplot2grid((14, 12), (10, 0), rowspan=2, colspan=12)
    ax2.plot(label1_start.index, np.ones(label1_start.index.shape[0]))
    for t, label in zip(label1_end.index, label1_end.values):
        if label == 'equ' or label == 'ne':
            ax2.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax2.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax2.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax2.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax2.axvline(t, color='y', linewidth=lw)
        if label == 'e':
            ax2.axvline(t, color='c', linewidth=lw)
    ax2.axes.get_yaxis().set_visible(False)
    plt.xlim([label1_start.index[0], label1_end.index[-1]])
    ax3 = plt.subplot2grid((14, 12), (12, 0), rowspan=2, colspan=12)
    ax3.plot(label2_start.index, np.ones(label2_start.index.shape[0]))

    for t, label in zip(label2_end.index, label2_end.values):
        if label == 'equ' or label == 'ne':
            ax3.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax3.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax3.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax3.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax3.axvline(t, color='y', linewidth=lw)
        if label == 'e':
            ax3.axvline(t, color='c', linewidth=lw)
    ax3.axes.get_yaxis().set_visible(False)
    # ax2.axes.get_xaxis().set_visible(False)
    plt.xlim([label1_start.index[0], label1_end.index[-1]])
    plt.savefig(figures_path / out_dir / file.stem)
    if show:
        plt.show()
    f.clear()
    plt.close(f)


def view_raw_file(file=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    if not file:
        file = random.choice(os.listdir('../data/raw/'))

    logger.info('reading file: ' + file)

    size_dist_df = pd.read_hdf('../data/raw/' + file, key='obs/particle')

    dndlogdp_df = cm3_to_dndlogdp(size_dist_df)

    plt.pcolor(size_dist_df.index,  # X axis
               [float(r) * 1e9 for r in size_dist_df.columns],  # Y axis
               (np.log10(np.absolute(np.asarray(dndlogdp_df)) + 10))[::1, ::1].T,  # Z axis
               cmap='jet')

    plt.yscale('log')


def evaluate_hyperparameter_for_label_debug(fi, kwargs):
    labels = get_labels_ene(fi, kwargs)
    size_dist_df = pd.read_hdf('../data/raw/' + fi, key='obs/particle')
    # dndlogdp_df = cm3_to_dndlogdp(size_dist_df)

    aux_df = pd.read_hdf('../data/raw/' + fi, key='aux')
    aux_df = aux_df.resample('10T').sum()
    delta_N_T = aux_df[['dn_nuc', 'dn_coa', 'dn_dep']].abs().sum(axis=1)
    delta_V_T = aux_df[['dvol_nuc', 'dvol_con', 'dvol_dep']].abs().sum(axis=1)

    plt_dim = (28, 12)

    ax1 = plt.subplot2grid(plt_dim, (0, 0), rowspan=10, colspan=12)

    plt.pcolor(size_dist_df.values[::1, ::1].T,  # Z axis
               cmap='jet')

    # plt.yscale('log')

    # cbar = plt.colorbar(ticks=[1, 2, 3, 4, 5])
    # plt.clim(1, 5)
    # cbar.ax.set_xticklabels(['10', '100', '100', '1000', '10000'])
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = plt.subplot2grid(plt_dim, (10, 0), rowspan=2, colspan=12)
    ax2.plot(labels.index, np.ones(labels.index.shape[0]))

    lw = 3

    for t, label in zip(labels.index, labels.values):
        if label == 'equ':
            ax2.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax2.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax2.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax2.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax2.axvline(t, color='y', linewidth=lw)

    ax2.axes.get_yaxis().set_visible(False)
    # ax2.axes.get_xaxis().set_visible(False)
    plt.xlim([labels.index[0], labels.index[-1]])
    ax2.xaxis.set_major_formatter(DateFormatter("%H"))

    ax3 = plt.subplot2grid(plt_dim, (12, 0), rowspan=8, colspan=12)
    ax3.plot(aux_df['dn_nuc'].abs() / delta_N_T, color='g')
    ax3.plot(aux_df['dn_coa'].abs() / delta_N_T, color='b')
    ax3.plot(aux_df['dn_dep'].abs() / delta_N_T, color='y')
    plt.xlim([labels.index[0], labels.index[-1]])

    ax4 = plt.subplot2grid(plt_dim, (20, 0), rowspan=8, colspan=12)
    ax4.plot(aux_df['dvol_nuc'].abs() / delta_V_T, color='g')
    ax4.plot(aux_df['dvol_con'].abs() / delta_V_T, color='r')
    ax4.plot(aux_df['dvol_dep'].abs() / delta_V_T, color='y')
    plt.xlim([labels.index[0], labels.index[-1]])


def raw_file(fi):
    size_dist_df = pd.read_hdf('../data/raw/malte-uhma/' + fi, key='obs/particle')

    plt_dim = (12, 12)

    plt.pcolor(size_dist_df.values[::1, ::1].T / 1e6,  # Z axis
               cmap='jet')
    plt.colorbar()
    plt.show()


def raw_file_as_dndlogdp(fi):
    size_dist_df = pd.read_hdf('../data/raw/malte-uhma/' + fi, key='obs/particle')

    dndlogdp_df = cm3_to_dndlogdp(size_dist_df / 1e6)

    plt.pcolor(dndlogdp_df[::1, ::1].T,  # Z axis
               cmap='jet')
    plt.colorbar()
    plt.show()


# def file(X, fi=None):
#     if fi is None:
#         files = open(X['script_file']).read().splitlines()
#         fi = random.choice(files)
#
#     _, obs, delta, acc = read_data(fi)
#
#     plt.pcolor(obs.values[::1, ::1].T,  # Z axis
#                cmap='jet')
#     plt.colorbar()
#     plt.show()


def sample(fi=None):
    if fi is None:
        fi = random.choice(os.listdir('../data/interim/test_D_A/'))

    _, obs, delta, acc = read_data(fi)

    plt.plot(obs.iloc[sample])
    plt.show()
    plt.plot(delta.iloc[sample])
    plt.show()
    plt.plot(acc.iloc[sample])
    plt.show()


# def testDA_file(fi=None, sample=None):
#     if fi is None:
#         fi = random.choice(os.listdir('../data/interim/test_D_A/'))
#
#     _, obs, delta, acc = read_data(fi)
#
#     plt.pcolor(obs.values[::1, ::1].T,  # Z axis
#                cmap='jet')
#     plt.colorbar()
#     plt.show()
#
#     if sample is not None:
#         plt.plot(obs.iloc[sample])
#         plt.show()
#         plt.plot(delta.iloc[sample])
#         plt.show()
#         plt.plot(acc.iloc[sample])
#         plt.show()
#
#
# def trainDA_file(fi=None, sample=None):
#     if fi is None:
#         fi = random.choice(os.listdir('../data/interim/train_D_A/'))
#
#     _, obs, delta, acc = read_data(fi, which='interim/train_D_A')
#
#     plt.pcolor(obs.values[::1, ::1].T,  # Z axis
#                cmap='jet')
#     plt.colorbar()
#     plt.show()
#
#     if sample is not None:
#         plt.plot(obs.iloc[sample])
#         plt.show()
#         plt.plot(delta.iloc[sample])
#         plt.show()
#         plt.plot(acc.iloc[sample])
#         plt.show()


def real_file(fi=None, sample=None):
    if fi is None:
        fi = random.choice(os.listdir('../data/external'))

    _, obs, delta, acc = read_data(fi, which='external/')

    plt.pcolor(obs.values[::1, ::1].T,  # Z axis
               cmap='jet')
    plt.colorbar()
    plt.show()

    if sample is not None:
        plt.plot(obs.iloc[sample])
        plt.show()
        plt.plot(delta.iloc[sample])
        plt.show()
        plt.plot(acc.iloc[sample])
        plt.show()


def real_file_with_label(fi=None, sample=None):
    if fi is None:
        fi = random.choice(os.listdir('../data/external'))

    _, obs, delta, acc = read_data( fi + '_D_A', which='external')

    label_start, label_end, labels = read_mlf_label('../data/interim/results.mlf', fi)
    # label_start, label_end, labels = read_mlf_label('../data/raw/dmps/manual_labels.mlf', fi)

    plt_dim = (14, 12)

    ax1 = plt.subplot2grid(plt_dim, (0, 0), rowspan=10, colspan=12)

    plt.pcolor(obs.values[::1, ::1].T,  # Z axis
               cmap='jet')
    # plt.colorbar()

    ax1.axes.get_xaxis().set_visible(False)

    ax2 = plt.subplot2grid(plt_dim, (10, 0), rowspan=2, colspan=12)
    ax2.plot(label_start.index, np.ones(label_start.index.shape[0]))

    lw = 3

    for t, label in zip(label_end.index, label_end.values):
        if label == 'equ' or label == 'ne':
            ax2.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax2.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax2.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax2.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax2.axvline(t, color='y', linewidth=lw)
        if label == 'e':
            ax2.axvline(t, color='c', linewidth=lw)

    plt.xlim([0, 144])

    plt.show()

    if sample is not None:
        plt.plot(obs.iloc[sample])
        plt.show()
        plt.plot(delta.iloc[sample])
        plt.show()
        plt.plot(acc.iloc[sample])
        plt.show()


def evaluate_hyperparameter_for_label(fi, hyperparameters):
    labels = get_labels_ene(fi, hyperparameters)
    size_dist_df = pd.read_hdf('../data/raw/' + fi, key='obs/particle')
    dndlogdp_df = cm3_to_dndlogdp(size_dist_df / 1e6)

    aux_df = pd.read_hdf('../data/raw/' + fi, key='aux')
    aux_df = aux_df.resample('10T').sum()
    delta_N_T = aux_df[['dn_nuc', 'dn_coa', 'dn_dep']].abs().sum(axis=1)
    delta_V_T = aux_df[['dvol_nuc', 'dvol_con', 'dvol_dep']].abs().sum(axis=1)

    plt_dim = (12, 12)

    ax1 = plt.subplot2grid(plt_dim, (0, 0), rowspan=10, colspan=12)

    plt.pcolor(size_dist_df.index,  # X axis
               [float(r) * 1e9 for r in size_dist_df.columns],  # Y axis
               (np.log10(np.absolute(np.asarray(dndlogdp_df)) + 10))[::1, ::1].T,  # Z axis
               cmap='jet')

    plt.yscale('log')

    # cbar = plt.colorbar(ticks=[1, 2, 3, 4, 5])
    plt.clim(0, 4)
    # cbar.ax.set_xticklabels(['10', '100', '100', '1000', '10000'])
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = plt.subplot2grid(plt_dim, (10, 0), rowspan=2, colspan=12)
    ax2.plot(labels.index, np.ones(labels.index.shape[0]))

    lw = 3

    for t, label in zip(labels.index, labels.values):
        if label == 'equ' or label == 'ne':
            ax2.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax2.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax2.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax2.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax2.axvline(t, color='y', linewidth=lw)
        if label == 'e':
            ax2.axvline(t, color='c', linewidth=lw)

    ax2.axes.get_yaxis().set_visible(False)
    # ax2.axes.get_xaxis().set_visible(False)
    plt.xlim([labels.index[0], labels.index[-1]])
    ax2.xaxis.set_major_formatter(DateFormatter("%H"))


def evaluate_results(fi=None):
    if fi is None:
        fi = random.choice(os.listdir(interim_data_path / 'test_D_A'))

    result_start, result_end, results, score = read_result_label(RESULTS_MLF_PATH, fi)
    label_start, label_end, labels = read_mlf_label(interim_data_path / 'labels.mlf', fi)

    _, size_dist_df, delta, acc = read_data(fi)

    # dndlogdp_df = cm3_to_dndlogdp(size_dist_df)

    # rads = pd.read_hdf('../data/raw/'+ data_version + '.' + fi + '.h5', key='obs/particle').columns
    plt_dim = (14, 12)

    ax1 = plt.subplot2grid(plt_dim, (0, 0), rowspan=10, colspan=12)

    # plt.pcolor(size_dist_df.index,  # X axis
    #            [float(r) * 1e9 for r in rads],  # Y axis
    #            (np.log10(np.absolute(np.asarray(dndlogdp_df)) + 10))[::1, ::1].T,  # Z axis
    #            cmap='jet')

    plt.pcolor(np.asarray(size_dist_df)[::1, ::1].T,  # Z axis
               # plt.pcolor(np.log10(np.absolute(np.asarray(size_dist_df)+10))[::1, ::1].T,  # Z axis
               cmap='jet')

    # plt.yscale('log')
    plt.title(fi)
    # cbar = plt.colorbar()
    plt.clim(0, 4)
    # cbar.ax.set_xticklabels(['10', '100', '100', '1000', '10000'])
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = plt.subplot2grid(plt_dim, (10, 0), rowspan=2, colspan=12)
    ax2.plot(label_start.index, np.ones(label_start.index.shape[0]))

    lw = 3

    for t, label in zip(label_end.index, label_end.values):
        if label == 'equ' or label == 'ne':
            ax2.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax2.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax2.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax2.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax2.axvline(t, color='y', linewidth=lw)
        if label == 'e':
            ax2.axvline(t, color='c', linewidth=lw)
    ax2.axes.get_yaxis().set_visible(False)
    # ax2.axes.get_xaxis().set_visible(False)
    plt.xlim([result_end.index[0], result_end.index[-1]])

    ax3 = plt.subplot2grid(plt_dim, (12, 0), rowspan=2, colspan=12)
    ax3.plot(result_start.index, np.ones(result_start.index.shape[0]))

    lw = 3

    for t, label in zip(result_end.index, result_end.values):
        if label == 'equ' or label == 'ne':
            ax3.axvline(t, color='k', linewidth=lw)
        if label == 'nuc':
            ax3.axvline(t, color='g', linewidth=lw)
        if label == 'con':
            ax3.axvline(t, color='r', linewidth=lw)
        if label == 'coa':
            ax3.axvline(t, color='b', linewidth=lw)
        if label == 'dep':
            ax3.axvline(t, color='y', linewidth=lw)
        if label == 'e':
            ax3.axvline(t, color='c', linewidth=lw)

    ax3.axes.get_yaxis().set_visible(False)
    # ax2.axes.get_xaxis().set_visible(False)
    plt.xlim([result_end.index[0], result_end.index[-1]])

    plt.show()


def read_result_label(mlf, date):
    with open(mlf, 'rt') as fi:
        if fi.readline() != '#!MLF!#\n':
            print('Not a MLF file')

        start = []
        end = []
        label = []
        score = []

        tmp_line = fi.readline().splitlines()[0]
        while not (tmp_line.endswith(date + '.rec"') or tmp_line.endswith(date + '.lab"')):
            tmp_line = fi.readline().splitlines()[0]

        tmp_line = fi.readline().splitlines()[0]
        while not tmp_line.endswith('.'):
            ts, te, l, s = tmp_line.splitlines()[0].split(' ')
            start.append(int(ts))
            end.append(int(te))
            label.append(l)
            score.append(float(s))
            tmp_line = fi.readline().splitlines()[0]

    start_df = pd.DataFrame(index=(np.array(start) / 10 / 60), data=label)
    end_df = pd.DataFrame(index=(np.array(end) / 10 / 60), data=label)

    start_df = start_df.reindex(range(0, 144), method='ffill')
    end_df = end_df.reindex(range(1, 145), method='bfill')

    return start_df, end_df, label, score


def read_mlf_label(mlf, date, file_length=None):
    with open(mlf, 'rt') as fi:
        if fi.readline() != '#!MLF!#\n':
            print('Not a MLF file')

        start = []
        end = []
        label = []

        tmp_line = fi.readline().splitlines()[0]
        while not (tmp_line.endswith(date + '.rec"') or tmp_line.endswith(date + '.lab"')):
            tmp_line = fi.readline().splitlines()[0]

        tmp_line = fi.readline().splitlines()[0]
        while not tmp_line.endswith('.'):
            tmp = tmp_line.splitlines()[0].split(' ')
            start.append(int(tmp[0]))
            end.append(int(tmp[1]))
            label.append(tmp[2])
            tmp_line = fi.readline().splitlines()[0]

    start_df = pd.DataFrame(index=(np.array(start) / 10 / 60), data=label)
    end_df = pd.DataFrame(index=(np.array(end) / 10 / 60), data=label)
    try:
        start_df = start_df.reindex(range(0, file_length), method='ffill')
        end_df = end_df.reindex(range(1, file_length+1), method='bfill')
    except ValueError:
        print('Error in label of date: ' + date)

    return start_df, end_df, label


if __name__ == '__main__':

    labels = {'mlf': '../data/interim/labels.mlf'}
    results = {'mlf': '../data/interim/results.mlf'}
    generate_plots(data, labels, results)
