import struct

import numpy as np
import pandas as pd

from src.models.HTK.htktools import NUMBER_OF_SIZE_BINS


def write_data(fo='', data=None):
    """
    :param fo:
    :param data:
    :return:
    """

    data = data.round(2)

    fo = open(fo, 'wb')

    _write_header(fo, data)

    # Guardo el DataFrame a binario.
    # Recorro cada linea

    for index, line in data.iterrows():
        # Recorro cada elemento de la linea
        for value in line.values:
            # Guardo el valor como binario
            fo.write(struct.pack('>f', value))


def read_data(fi):
    dt = np.dtype([
        ("nSamples", '>u4'),
        ("sampPeriod", '>u4'),
        ("sampSize", '>u2'),
        ("paramKind", '>u2'),
        #("data", '>f')
    ])

    # n is the number of vectors of data stored in HTK format. If only parameters are stored, n=1, if accelerations are
    # also stored, n=2, etc...
    n = 1
    # if which.find('D') != -1:
    #     n += 1
    # if which.find('A') != -1:
    #     n += 1
    # if which.find('T') != -1:
    #     n += 1

    with open(fi, 'rb') as data:
        header = np.fromfile(data, dtype=dt, count=1)
        data = np.fromfile(data, dtype=np.dtype([("data", '>f')]), count=int(header['nSamples']*n*NUMBER_OF_SIZE_BINS))

    data = pd.DataFrame.from_records(data['data'].reshape(int(header['nSamples']), n*NUMBER_OF_SIZE_BINS))

    return header, data.iloc[:, :25]  #, data.iloc[:, 26:50], data.iloc[:, 51:75]


def _write_header(fo, data):
    """
    This function writes the header of the HTK parameters binary file.
    """
    nSamples = data.__len__()  # Calculo la cantidad de muestras que va a contener el archivo.
    numberOfFeatures = data.columns.size  # Number of sizes of DMPS.
    sampPeriod = int(600)
    # HTK toma como si fuesen 600 us pero en realidad son 600 segundos (10 minutos). Esto se
    # se debe a que no es posible representar el periodo de 600 segundos con un
    # int de 4 bytes. Es decir que los tiempos se dividen por 1e6.
    sampSize = 4 * numberOfFeatures  # Tamaño del float
    parmKind = 9  # USER (user defined sample kind)

    fo.write(struct.pack('>i', nSamples))
    fo.write(struct.pack('>i', sampPeriod))
    fo.write(struct.pack('>h', sampSize))
    fo.write(struct.pack('>h', parmKind))
