import logging

import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

from scipy import integrate as integrate
from scipy.optimize import curve_fit, nnls
from scipy.special import erf


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

e = 1.602E-19
eo = 8.854e-12
boltz = 1.381e-23


def read_raw_dmps(fi):
    """ Reads raw data from directory ../data/raw and returns a pandas DataFrame

    """
    raw_data_path = '../../data/raw/dmps/raw/'

    logging.info('File name: ' + fi)

    year = fi[2:6]
    month = fi[6:8]
    day = fi[8:10]

    # Read odd rows (starting with first line)
    names_odd_rows = ['hour', 'minute', 'second',
                      'temp', 'press', 'hum', 'NA', 'excess', 'sample',
                      'voltage_1', 'voltage_2', 'voltage_3', 'voltage_4', 'voltage_5', 'voltage_6', 'voltage_7',
                      'voltage_8', 'voltage_9', 'voltage_10',
                      'voltage_11', 'voltage_12', 'voltage_13', 'voltage_14', 'voltage_15', 'voltage_16', 'voltage_17',
                      'voltage_18', 'voltage_19', 'voltage_20',
                      'voltage_21', 'voltage_22', 'voltage_23', 'voltage_24', 'voltage_25']

    data0 = pd.read_csv(raw_data_path + fi, sep='\t', skiprows=lambda x: x % 2 == 1, names=names_odd_rows)
    data0['year'] = year
    data0['month'] = month
    data0['day'] = day
    data0.index = pd.to_datetime(data0[['year', 'month', 'day', 'hour', 'minute', 'second']])

    names_even_rows = ['hour', 'minute', 'second',
                       'temp', 'press', 'hum', 'NA', 'excess', 'sample',
                       'concentration_1', 'concentration_2', 'concentration_3', 'concentration_4', 'concentration_5',
                       'concentration_6', 'concentration_7', 'concentration_8', 'concentration_9', 'concentration_10',
                       'concentration_11', 'concentration_12', 'concentration_13', 'concentration_14',
                       'concentration_15', 'concentration_16', 'concentration_17', 'concentration_18',
                       'concentration_19', 'concentration_20',
                       'concentration_21', 'concentration_22', 'concentration_23', 'concentration_24',
                       'concentration_25']

    data1 = pd.read_csv(raw_data_path + fi, sep='\t', skiprows=lambda x: x % 2 == 0, names=names_even_rows)
    data1['year'] = year
    data1['month'] = month
    data1['day'] = day
    data1.index = pd.to_datetime(data1[['year', 'month', 'day', 'hour', 'minute', 'second']])

    data = pd.concat([data0, data1], axis=1)
    data = data.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def invert(raw_data):
    tikh = 0  # If 1, tikhonov regularisation method(slow!)
    sigma = 1.0
    plkm = 6  # maximum number of charges in one particle

    # Getting DMPS-system information
    dma_values = get_dma_const()

    # Get Reynolds number
    reynolds_number = reynolds(dma_values['pipeflow'],
                               dma_values['pipediameter'],
                               raw_data['temp'].mean(),
                               raw_data['press'].mean())

    t = raw_data['temp'].mean()
    pr = raw_data['press'].mean()
    rh = raw_data['hum'].mean()
    flow = raw_data['excess'].mean()
    sample = raw_data['sample'].mean()

    logging.info('Reynolds number in sample line: ' + str(reynolds_number))
    # logging.info('File name: ' + filena(1:10))
    logging.info('Temperature:  ' + str(t - 273.15) + ' C')
    logging.info('Pressure:     ' + str(pr/100) + ' hPa')
    logging.info('RH:           ' + str(rh) + ' %')
    logging.info('Aerosol flow: ' + str(sample) + ' L/min')
    logging.info('Sheath flow:  ' + str(flow) + ' L/min')

    # First setvolt column in datafile
    apu = 10

    # Reading voltages
    voltages = get_daily_mean_voltages(raw_data)

    # creating size grid to start with
    dp = 10 ** np.arange(-10, -5 + 0.002, step=0.002)

    # calculating the mobilities in the peaks
    mob_peak, dp_peak = mob_peaks(voltages, dma_values)
    ldp = np.log10(dp_peak)

    # Final grid
    inv_data = raw_data

    # Calculate channel borders
    dp_limits = np.mean([np.append(np.nan, ldp.values), np.append(ldp.values, np.nan)], axis=0)
    dp_limits[0] = ldp[0] - (ldp[1] - ldp[0]) / 2
    dp_limits[-1] = ldp[-1] + (ldp[-1] - ldp[-2]) / 2
    dlogdp = np.diff(dp_limits)
    dp_limits = 10 ** dp_limits

    p = np.linspace(1, plkm, plkm) * dma_values['polarity']

    tee = np.zeros((voltages.__len__(), dp_peak.__len__()))
    # Integrate hetransfer functions(check: Stohlzenburg 1988)
    for ii in range(0, voltages.__len__()):
        for jj in range(0, dp_peak.__len__()):
            tee[ii, jj] = integrate.quadrature(intfun, np.log10(dp_limits[jj]), np.log10(dp_limits[jj + 1]),
                                               args=(t, pr, p, voltages[ii],
                                               dma_values['pituus1'], dma_values['arkaksi1'], dma_values['aryksi1'],
                                               dma_values['qa1'], dma_values['qc1'], dma_values['qm1'],
                                               dma_values['qs1'],
                                               dma_values['cpcmodel1'], dma_values['dmamodel1'],
                                               dma_values['pipelength'], dma_values['pipeflow']),
                                               miniter=15, maxiter=200)[0] \
                          / (np.log10(dp_limits[jj + 1]) - np.log10(dp_limits[jj]))

    # TODO: Complete options below
    pmatriisi = np.linalg.pinv(tee)

    m, n = inv_data.shape
    apu1 = 2

    kokoja = np.zeros((dma_values['volt1lkm'], m))
    totconc = np.zeros(m)

    for j in range(0, m):
        # conc(1: m / 2, 1: volt1lkm)=v(2: 2:m / 2, apu: apu + volt1lkm - 1);
        # conc = inv_data.iloc[apu1, apu:apu + dma_values['volt1lkm'] - 1]
        conc = inv_data.iloc[j][inv_data.columns[pd.Series(inv_data.columns).str.startswith('conc')]]
        kk1 = np.mean(conc[-1 - 1:])
        conc = conc - kk1

        # Multiplying the concentration with kernel(=from Stohlzenburg thesis) and dlogDp

        if tikh != 1:
            #kokojak = pmatriisi'*conc'. * (1. / parkoko(:, 2))
            # NOTE: tee=matriisi
            kokojak = nnls(tee.T, conc)[0] * (1. / dlogdp)

        # TODO: Complete option below
        # if tikh == 1:
        #     lambda_l = l_curve(U, s, conc)
        #     kokojak = tikhonov(U, s, V, conc,lambda_l) * (1 / parkoko(:,2))
        #     # modeljak = matriisi'*(kokojak .*parkoko(:,2));

        kokoja[:, j] = kokojak.copy()
        if tikh != 1:
            totconc[j] = sum(np.dot(pmatriisi, conc.values))
        # TODO: Complete option below
        # if tikh == 1:
        #     totconc[j] = sum(tikhonov(U, s, V, conc,lambda_l))

    #     % Creating and saving final result file
    #     [mk nk]=size(kokoja');
    #     result(2:mk + 1, 3: nk + 2)=kokoja
    #     ';
    #     result(1, 1: 2)=[0 0];
    #     result(2: mk + 1, 1)=time;
    #     result(2: mk + 1, 2)=totconc
    #     ';
    #     result(1, 3: nk + 2)=parkoko(:, 1)';

    return inv_data, kokoja, totconc, dp_peak


def intfun(dp, t, press, p, volt, pituus, arkaksi, aryksi, qa, qc, qm, qs, cpcmodel, dmamodel, pipelength, pipeflow):
    dporig = dp
    dp = 10. ** dp

    # Laminar flow tube losses
    tubeloss = ltubefl(dp, pipelength, pipeflow, t, press)

    # Losses in the CPC
    if cpcmodel == '3025':
        cpcloss = tsi3025(dp, t, press)
    if cpcmodel == '3010':
        cpcloss = tsi3010(dp)
    if cpcmodel == '3022':
        cpcloss = tsi3022(dp)

    # Losses in the DMA
    if dmamodel == 'HAUM':
        dmaloss = haukem(dp, qa, t, press)
    if dmamodel == 'HAUS':
        dmaloss = haukes(dp, qa, t, press)
    if dmamodel == 'TSIL':
        dmaloss = tsi3071(dp, qa, t, press)

    # All the losses  summed(diffusion losses equation)
    totalloss = (tubeloss * cpcloss * dmaloss)

    # "Triangles"(Stohlzenburg)
    tr = teearra(p, dp, t, press, volt, pituus, arkaksi, aryksi, qa, qc, qm, qs)
    # Charging efficiency(from: Wiedensohler 1989)!
    charge = varaus(dp, p, t)
    # Everything together
    res = (np.nansum((tr * charge), axis=1) * totalloss)
    dp = dporig

    return res


def teearra(pp, dp, t, p, voltage, pituus, arkaksi, aryksi, qa, qc, qm, qs):
    beta = (qs + qa) / (qm + qc)

    delta = -(qs - qa) / (qs + qa)

    gammas = (aryksi / arkaksi) * (aryksi / arkaksi)

    gkappa = pituus * arkaksi / (arkaksi * arkaksi - aryksi * aryksi)

    gammai = (0.25 * (1 - gammas * gammas) * (1 - gammas) * (1 - gammas) + (5 / 18) *
              (1 - gammas * gammas * gammas) * (1 - gammas) * np.log(gammas) + (1 / 12) * (
                      1 - gammas * gammas * gammas * gammas) *
              np.log(gammas) * np.log(gammas)) / (
                     (1 - gammas) * (-0.5 * (1 + gammas) * np.log(gammas) - (1 - gammas)) * (
                     -0.5 * (1 + gammas) * np.log(gammas) - (1 - gammas)))

    gabeta = (4.0 * (1 + beta) * (1 + beta) * (
            gammai + (1 / ((2 * (1 + beta) * gkappa) * (2 * (1 + beta) * gkappa))))) / (1 - gammas)

    zeta = np.zeros((dp.__len__(), pp.__len__()))

    # size(pp)
    # size(cunn(dp,t,p) ./dp)
    # size(pp*(cunn(dp,t,p) ./dp)')
    zeta = (pp * (e * cunn(dp, t, p)[:, np.newaxis] / (3 * np.pi * visc(t) * dp[:, np.newaxis])))
    zetap = np.zeros((dp.__len__(), pp.__len__()))

    # size(zeta)
    # pause

    zetap = 4.0 * voltage * np.pi * pituus * zeta / ((qm + qc) * np.log(aryksi / arkaksi))

    # semilogx(dp,zeta)
    # pause

    # semilogx(dp,zetap)
    # pause

    rhota = np.zeros((dp.__len__(), pp.__len__()))

    for i in range(0, pp.__len__()):
        rhota[:, i] = np.sqrt((zetap[:, i] / pp[i]) * (gabeta * np.log(aryksi / arkaksi) * boltz * t / (e * voltage)))

    # semilogx(dp,rhota)
    # pause

    teea1 = rhota / (np.sqrt(2) * beta * (1. - delta)) * (epsilon(abs(zetap - (1 + beta)) / (np.sqrt(2) * rhota)) +
                                                          epsilon(abs(zetap - (1 - beta)) / (np.sqrt(2) * rhota)) -
                                                          epsilon(
                                                              abs(zetap - (1 + beta * delta)) / (np.sqrt(2) * rhota)) -
                                                          epsilon(
                                                              abs(zetap - (1 - beta * delta)) / (np.sqrt(2) * rhota)))

    teea2 = 1 / (2 * beta * (1 - delta)) * (abs(zetap - (1 + beta)) +
                                            abs(zetap - (1 - beta)) -
                                            abs(zetap - (1 + beta * delta)) -
                                            abs(zetap - (1 - beta * delta)))
    teea = teea2 + teea1
    return teea


def epsilon(tuntea):
    return -tuntea * (1. - erf(tuntea)) + (1. / np.sqrt(np.pi)) * np.exp(-tuntea * tuntea)


def visc(t):
    return (174. + 0.433 * (t - 273.)) * 1.0e-7


def cunn(dp, t, p):
    return 1.0 + rlambda(t, p) / dp * (2.514 + 0.800 * np.exp(-0.55 * dp / rlambda(t, p)))


def rlambda(t, p):
    dm = 3.7e-10
    avoc = 6.022e23
    kaasuv = 8.3143

    return kaasuv * t / (np.sqrt(2.) * avoc * p * np.pi * dm * dm)


def varaus(dp, pp, t):
    # Charging distribution
    #  varaus = reservation

    alfa = np.zeros((2, 6))

    if pp[1] > 0:
        alfa[0, 0] = -2.3484
        alfa[0, 1] = 0.6044
        alfa[0, 2] = 0.4800
        alfa[0, 3] = 0.0013
        alfa[0, 4] = -0.1553
        alfa[0, 5] = 0.0320
        alfa[1, 0] = -44.4756
        alfa[1, 1] = 79.3772
        alfa[1, 2] = -62.89
        alfa[1, 3] = 26.4492
        alfa[1, 4] = -5.748
        alfa[1, 5] = 0.5049
    else:
        alfa[0, 0] = -2.3197
        alfa[0, 1] = 0.6175
        alfa[0, 2] = 0.6201
        alfa[0, 3] = -0.1105
        alfa[0, 4] = -0.1260
        alfa[0, 5] = 0.0297
        alfa[1, 0] = -26.3328
        alfa[1, 1] = 35.9044
        alfa[1, 2] = -21.4608
        alfa[1, 3] = 7.0867
        alfa[1, 4] = -1.3088
        alfa[1, 5] = 0.1051

    coeff = np.zeros((dp.__len__(), pp.__len__()))
    coefft = np.zeros((dp.__len__(), pp.__len__()))

    for i in range(1, 7):
        coefft[:, 0:min(2, pp.__len__())] = coefft[:, 0:min(2, pp.__len__())] + (
                alfa[0:min(2, pp.__len__()), i-1] * (np.log10(dp / 1e-9)[:, np.newaxis] ** (i - 1)))

    coeff[:, 0:min(2, pp.__len__())] = 10 ** coefft[:, 0:min(2, pp.__len__())]

    for i in range(2, pp.__len__()-1):
        coe = (2.0 * np.pi * eo * dp * boltz * t) / e ** 2
        coeff[:, i] = ((1.0 / np.sqrt(coe * 2.0 * np.pi)) * np.exp(
            -(-np.sign(pp[0]) * i - coe * 0.1335) ** 2 / (2.0 * coe)))

    return coeff


def haukem(dpp, aeroflow, temp, press):
    # This is actually unknown
    plength = 5
    res = ltubefl(dpp, plength, aeroflow, temp, press)
    # res=0.85-exp((-1e9*dpp/7.5))'
    # res=1-exp((-1e9*dpp/7.5))'

    # mob=e*cunn(dpp,temp,press) ./(3*pi*visc(temp)*dpp)

    # %elecl=exp(-10e5*mob)'

    # %res=ltubefl(dpp,0.0,aeroflow,temp,press)'
    res = ltubefl(dpp, 4.7, aeroflow, temp, press)
    # %dp=dpp*1e9
    # %res=0.85-exp(-dp/8)'
    # res=ltubefl(dpp,27.2,aeroflow,temp,press)'.*elecl
    # semilogx(dpp,res)

    return res


def haukes():
    # TODO: Complete
    return


def tsi3071():
    # TODO: Complete
    return


def tsi3025():
    # TODO: Complete
    return


def tsi3022():
    # TODO: Complete
    return


def tsi3010(dp):
    # S. Mertes et. al.
    # Aerosol Science and Technology 23:257:261 (1995)

    # Lämpötilaero

    TD = 25
    if TD != 25 & TD != 21 & TD != 17 & TD != 1:
        TD = 17

    if TD == 25:
        a = 1.7
        D1 = 4.3
        D2 = 1.5
        DP50 = 5.7

    if TD == 21:
        a = 1.4
        D1 = 6.5
        D2 = 1.9
        DP50 = 7.6

    if TD == 17:
        a = 1.4
        D1 = 8.9
        D2 = 2.9
        DP50 = 10.5

    if TD == 1:
        a = 101.07283
        b = 297.7012
        c = 1.83913
        d = 2.98868

    Dpp = 1e9 * dp
    D0 = D2 * np.log(a - 1) + D1
    res = np.zeros(Dpp.__len__())
    for i in range(0, Dpp.__len__()):
        # res(i)=1
        if Dpp[i] >= D0:
            res[i] = 1 - a * (1 + np.exp((Dpp[i] - D1) / D2)) ** (-1)
        # res1 = a - b./(1.0 + exp((Dpp-c)./d))
        # res=res1./100
        else:
            res[i] = 0

    return res


def ltubefl(dpp, plength, pflow, temp, press):
    rmuu = np.pi * diffuus(dpp, temp, press) * plength / pflow
    res = np.zeros(dpp.__len__())
    for i in range(0, dpp.__len__()):
        if rmuu[i] < 0.02:
            res[i] = 1 - 2.56 * rmuu[i] ** (2 / 3) + 1.2 * rmuu[i] + 0.177 * rmuu[i] ** (4 / 3)
        else:
            res[i] = 0.819 * np.exp(-3.657 * rmuu[i]) + 0.097 * np.exp(-22.3 * rmuu[i]) + 0.032 * np.exp(-57 * rmuu[i])

    return res


def diffuus(dpp, temp, press):
    K = 1.38e-23
    return (K * temp * cunn(dpp, temp, press)) / (3 * np.pi * visc(temp) * dpp)


def mob_peaks(voltages, dma_values):
    mob_peak1 = ((dma_values['qc1'] + dma_values['qm1']) / 2) * np.log(
        dma_values['arkaksi1'] / dma_values['aryksi1']) / (2 * np.pi * dma_values['pituus1']) * 1. / voltages

    # calculating the diameters in the peaks
    dp_peak = min_mob(mob_peak1, dma_values['tem'], dma_values['press'])

    return mob_peak1, dp_peak


def min_mob(mob, t, p):
    dp = 1e-9 * np.ones(mob.__len__())

    dpt = np.ones(mob.__len__())

    while max(abs(dp - dpt) / dpt) > 1e-6:
        dp = dpt
        ee = 1.602E-19
        dpt = ee * cunn(dp, t, p) / (3 * np.pi * visc(t) * mob)  # tulos

    return dpt