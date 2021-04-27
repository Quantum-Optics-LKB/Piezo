# -*- coding: utf-8 -*-

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter, find_peaks, correlate
from classspectrum import DisplaySpectrum

ds = DisplaySpectrum()

def read_ch(csv_name):

    time = []
    ch1  = []
    ch2  = []
    ch3  = []

    with open(csv_name+'.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            try :
                time.append(float(row[0]))
                ch1.append(float(row[1]))
                ch2.append(float(row[2]))
                #ch3.append(float(row[3]))
            except:
                pass

    time = np.array(time)
    ch = [ch1, ch2]

    for i in range (0, len(ch)) :
        ch[i] = np.array(ch[i])
        #ch[i] = ch[i]-np.mean(ch[i][:100])
        #ch[i] = ch[i]/np.max(ch[i])
        ch[i] = savgol_filter(ch[i], 4001, 3)

    return time, ch


def peak(piezo, abso):

    maxi = find_peaks(abso, distance=10000, width=3000)[0]

    arg_1  = np.argmin(abso[maxi])
    mini_1 = maxi[arg_1]

    maxi = np.concatenate((maxi[:arg_1], maxi[arg_1+1:]))

    arg_2  = np.argmin(abso[maxi])
    mini_2 = maxi[arg_2]

    #plt.plot(piezo, abso)
    #plt.plot(piezo[mini_1], abso[mini_1], 'o')
    #plt.plot(piezo[mini_2], abso[mini_2], 'o')
    #plt.show()

    delta_v = piezo[mini_2] - piezo[mini_1]
    orig    = piezo[mini_1]

    return orig, delta_v


def piezo_to_hz(orig, delta_v, temps):

    #delta_v <-> 3.036 GHz
    #print(delta_v)
    temps = (temps-orig)*3.036/delta_v

    return temps


def fit_temp(param, ch=[0]):
    T = param[0]
    trans = ds.transmission(99.5, T, 0.1)
    trans = trans/np.max(trans)

    if len(trans) < len(ch):
        ratio = len(ch)//len(trans)
        ch = ch[::ratio]
        ch = ch[:len(trans)]

    diff = []
    for i in range (0, len(trans)):
        if ch[i]<0.94:
            diff.append(trans[i]-ch[i])
        else :
            diff.append(0)
    diff = np.array(diff)

    return np.sum(diff**2)


def fit_temp_2(temps, T):
    return ds.transmission(99.5, T, 0.1)


if __name__=="__main__":

    path = str(input('Nom du dossier : '))
    name = str(input('Nom du fichier csv : '))
    temps, ch_tot = read_ch(str(path)+'/'+str(name))
    orig, delta_v = peak(temps, ch_tot[0])
    temps = piezo_to_hz(orig, delta_v, temps)

    debut = np.nonzero(np.where(temps >= -6+1.308, 1, 0))[0][0]
    fin = np.nonzero(np.where(temps >= 6+1.308, 1, 0))[0][0]

    temps = temps[debut:fin] - 1.308

    for i in range(0, len(ch_tot)):
        ch_tot[i] = ch_tot[i][debut:fin]
        ch_tot[i] = ch_tot[i] - np.min(ch_tot[i])
        #ch_tot[i] = ch_tot[i] - 0.285
        ch_tot[i] = ch_tot[i]/np.max(ch_tot[i])
        #plt.plot(temps, ch_tot[i])
        #plt.show()

    np.savetxt(path+'/'+'temps_{}.txt'.format(name), np.column_stack((temps)))
    np.savetxt(path+'/'+'ch_{}.txt'.format(name), np.column_stack((ch_tot[1])))
    print('Fichiers txt sauvegardés')

    temps = np.loadtxt(path+'/'+'temps_{}.txt'.format(name))
    ch    = np.loadtxt(path+'/'+'ch_{}.txt'.format(name))

    res = minimize(fit_temp, x0=[415], args=ch)

    trans = ds.transmission(99.5, res.x, 0.1)

    if len(trans) < len(ch):
        ratio = len(ch)//len(trans)
        ch = ch[::ratio]
        temps = temps[::ratio]
        ch = ch[:len(trans)]
        temps = temps[:len(trans)]

    #ptot, pcov = curve_fit(fit_temp_2, temps, ch, p0=[315], bounds=([270], [500]))

    plt.plot(temps + 2.4446, trans, label='Theorie : {:.2f}K'.format(res.x[0]))
    plt.plot(temps + 2.4446, ch*np.max(trans), label='Experience')
    plt.xlabel('Detuning (GHz)')
    plt.ylabel('Transmission')
    plt.legend()
    plt.show()
