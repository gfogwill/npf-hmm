import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from src.data.htk import write_data
import scipy.integrate as int
from scipy.special import erf

e = 1.602E-19
eo = 8.854e-12
boltz = 1.381e-23


def read_raw_dmps(fi):
    """ Reads raw data from directory ../data/raw and returns a pandas DataFrame

    """
    raw_data_path = '../../data/raw/dmps/raw/'

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
            tee[ii, jj] = int.quadrature(intfun, np.log10(dp_limits[jj]), np.log10(dp_limits[jj + 1]),
                                         args=(t, pr, p, voltages[ii],
                                               dma_values['pituus1'], dma_values['arkaksi1'], dma_values['aryksi1'],
                                               dma_values['qa1'], dma_values['qc1'], dma_values['qm1'],
                                               dma_values['qs1'],
                                               dma_values['cpcmodel1'], dma_values['dmamodel1'],
                                               dma_values['pipelength'], dma_values['pipeflow']),
                                         miniter=15, maxiter=200)[0] \
                          / (np.log10(dp_limits[jj + 1]) - np.log10(dp_limits[jj]))

    pmatriisi = np.linalg.pinv(tee)
    return inv_data


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
    res = (np.nansum((tr * charge.transpose()), axis=0) * totalloss)
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
    zeta = (pp[:, np.newaxis] * (e * cunn(dp, t, p) / (3 * np.pi * visc(t) * dp)))
    zetap = np.zeros((dp.__len__(), pp.__len__()))

    # size(zeta)
    # pause

    zetap = 4.0 * voltage * np.pi * pituus * zeta / ((qm + qc) * np.log(aryksi / arkaksi))

    # semilogx(dp,zeta)
    # pause

    # semilogx(dp,zetap)
    # pause

    rhota = np.zeros((pp.__len__(), dp.__len__()))

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

    for i in range(1, 6):
        coefft[:, 1:min(2, pp.__len__())] = coefft[:, 1:min(2, pp.__len__())] + (
                alfa[1:min(2, pp.__len__()), i] * (np.log10(dp / 1e-9)[:, np.newaxis] ** (i - 1)))

    coeff[:, 1:min(2, pp.__len__())] = 10 ** coefft[:, 1:min(2, pp.__len__())]

    for i in range(3, pp.__len__()):
        coe = (2.0 * np.pi * eo * dp * boltz * t) / e ** 2
        coeff[:, i] = ((1.0 / np.sqrt(coe * 2.0 * np.pi)) * np.exp(
            -(-np.sign(pp[1]) * i - coe * 0.1335) ** 2 / (2.0 * coe)))

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


def cunn(dp, t, p):
    c = 1.0 + rlambda(t, p) / dp * (2.514 + 0.800 * np.exp(-0.55 * dp / rlambda(t, p)))
    return c


def rlambda(t, p):
    dm = 3.7e-10
    avoc = 6.022e23
    kaasuv = 8.3143

    r = kaasuv * t / (np.sqrt(2.) * avoc * p * np.pi * dm * dm)
    return r


def get_daily_mean_voltages(data):
    """ Get daily average voltage for each channel.

    Returns positive voltages
    """

    return data.loc[:, data.columns.str.startswith('voltage')].mean().abs()


def reynolds(pipeflow, pipediameter, t, pr):
    density = 1.29 * (273 / t) * (pr / 101325)
    pipearea = np.pi / 4 * pipediameter ** 2

    velocity = pipeflow / pipearea
    rey = density * velocity * pipediameter / visc(t)

    return rey


def visc(t):
    return (174. + 0.433 * (t - 273.)) * 1.0e-7


def get_dma_const():
    # This is used after 11.3.2003 at Utö

    # Change the system information here
    dma_const = dict(tem=295.15, press=1.00e5, rh=999.99, dmalkm=1, volt1lkm=25, polarity=1, pipelength=2.0,
                     pipeflow=1 / 1000 / 60, pipediameter=4 / 1000, pituus1=0.28, arkaksi1=3.3e-2, aryksi1=2.5E-2,
                     dmamodel1='HAUM', cpcmodel1='3010', qc1=5.0 / 1000 / 60, qm1=5.0 / 1000 / 60, qa1=1.0 / 1000 / 60,
                     qs1=1.0 / 1000 / 60)
    return dma_const


if __name__ == '__main__':
    data = read_raw_dmps('DM20170101.DAT')
    inv_data = invert(data)
