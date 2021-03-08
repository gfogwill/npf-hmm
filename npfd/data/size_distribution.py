import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit


TEST_REAL_FILES = ['DM20170507.cle',
                   'DM20170508.cle',
                   'DM20170523.cle',
                   'DM20170524.cle',
                   'DM20170525.cle',
                   'DM20171110.cle',
                   'DM20171113.cle',
                   'DM20171115.cle']


def decimalDOY2datetime(dDOY, year=2017):
    """"
    Esta funcion convierte la fecha de formato DOY con decimales a formato 'datetime'.
    """
    epoch = datetime(year - 1, 12, 31)

    # list(map(float, dDOY)) -> Convierto la la lista de str's a lista de float's.
    # map(lambda x: epoch+timedelta(days=x) -> convierto el DOY decimal a dateime
    try:
        result = list(map(lambda x: epoch + timedelta(days=x), list(map(float, dDOY))))
    except TypeError:
        result = epoch + timedelta(days=dDOY)

    return result


def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2 / 2 / sigma ** 2)


def lognorm(x, mu, sigma, A):
    return (A/np.sqrt(2*np.pi)/np.log(sigma))*np.exp(-np.log(x/mu)**2/2/np.log(sigma)**2)


def bilognorm(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return lognorm(x, mu1, sigma1, A1)+lognorm(x, mu2, sigma2, A2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)


def make_stats(window=2):
    stats = pd.DataFrame(index=pd.date_range(datetime(2017, 1, 1), datetime(2018, 1, 1)))

    dmps_data = pd.DataFrame()

    rads = [1.0000111e-08, 1.2003355e-08, 1.4407894e-08, 1.729391e-08, 2.0757897e-08, 2.4916266e-08, 2.9907466e-08,
             3.5898359e-08, 4.3089797e-08, 5.172031e-08, 6.2080165e-08, 7.4515032e-08, 8.9441755e-08, 1.0735904e-07,
             1.2886439e-07, 1.5467883e-07, 1.8566246e-07, 2.228529e-07, 2.6749686e-07, 3.2108028e-07, 3.8539677e-07,
             4.6259638e-07, 5.5526079e-07, 6.664882e-07, 7.9999642e-07]
    x = np.linspace(0, 24, 25)

    with open('../../data/external/marambio_stats.csv', 'wt') as fo:
        fo.write('date,mu1, sigma1, A1 mu2, sigma2, A2\n')
        for day in pd.date_range(start='01/01/2017', end='31/12/2017'):
            for i in range(window):
                file_names_mask = (day + timedelta(int(window/2-window+i))).strftime('%m%d')

                for file in os.listdir('../data/raw/dmps/inv/'):
                    if file.endswith(file_names_mask + '.cle'):
                        tmp_data = read_inv_file(file)
                        tmp_data.columns = rads
                        dmps_data = pd.concat([dmps_data, tmp_data])

            expected = (30e-9, 1.3, 5e7, 200e-9, 1.3, 1e7)

            y = dlog_to_dn(dmps_data.mean(axis=0))
            params, cov = curve_fit(bilognorm, rads, y.values, expected)
            sigma = np.sqrt(np.diag(cov))
            params_str = np.array2string(params, separator=',', precision=2, ).replace('[', '').replace(']', '')
            fo.write(day.strftime('%Y-%m-%d,') + params_str + '\n')
            # pd.DataFrame(data={'params': params, 'sigma': sigma}, index=bimodal.__code__.co_varnames[1:])

            if True:
                import matplotlib.pyplot as plt
                plt.plot(rads, y)
                plt.plot(rads, bilognorm(rads, *params), color='red', lw=3, label='model')
                plt.xscale('log')
                plt.show()
    return


def convert_inverted_file_to_htk(fi):
    tmp_data = read_inv_file(fi)


def read_inv_file(fi):
    tmp_data = pd.read_csv('../data/raw/dmps/inv/' + fi, sep=r'\s+')
    tmp_data = tmp_data.replace(np.nan, -1)
    tmp_data.index = tmp_data.iloc[:, 0].apply(decimalDOY2datetime)
    tmp_data = tmp_data.drop(columns=tmp_data.columns[[0, 1]]).resample('10T').ffill()
    tmp_data = tmp_data.replace(np.nan, 0)
    tmp_data.columns = [float(x) for x in tmp_data.columns]
    tmp_data.index.name = 'datetime'

    return tmp_data


def dlog_to_dn(conc_dmps):
    rdry_meas = conc_dmps.index
    sections_meas = conc_dmps.shape[0]
    middlepoints = np.zeros(sections_meas + 1)

    # Calculate bin middlepoints
    for jj in range(1, sections_meas):
        middlepoints[jj] = rdry_meas[jj - 1] + rdry_meas[jj]

    middlepoints[0] = 4. * rdry_meas[0] - middlepoints[1]
    middlepoints[sections_meas] = 4. * rdry_meas[sections_meas - 1] - middlepoints[sections_meas - 1]
    middlepoints = np.log10(middlepoints)

    dp = np.diff(middlepoints)

    conc_dmps = conc_dmps * dp * 1.e6 + 1.  # cm^-3 -> m^-3

    return conc_dmps


def cm3_to_dndlogdp(size_dist_df):
    """ Change malte data from N/cm3 to dN/dlogDp
    """
    num_par = size_dist_df.values
    radius = [float(r)*2 for r in size_dist_df.columns]

    fx, fy = num_par.shape
    pnum3 = np.zeros((fx, fy))

    for i in range(0, fx):
        for j in range(1, fy - 2):
            pnum3[i, j] = num_par[i, j] / ((np.log10(radius[j + 1]) - np.log10(radius[j])) / 2
                                           + (np.log10(radius[j]) - np.log10(radius[j - 1])) / 2)
        pnum3[i, fy - 1] = num_par[i, fy - 1] / (np.log10(radius[j]) - np.log10(radius[j - 1]))
        pnum3[i, 0] = num_par[i, 0] / (np.log10(radius[j]) - np.log10(radius[j - 1]))

    return pnum3


if __name__ == '__main__':
    make_stats()


