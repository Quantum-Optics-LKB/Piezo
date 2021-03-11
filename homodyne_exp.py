import numpy as np
import cv2
import EasyPySpin
import matplotlib.pyplot as plt
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
from Homodyne import Homodyne
import os
import sys

plt.switch_backend('Qt5Agg')
plt.ion()

# piezo = PiezoTIM101('65863991')
# piezo.move_to(channel=1, pos=-100)
# piezo.move_to(channel=1, pos=0)
# piezo.disconnect()

cam = EasyPySpin.VideoCapture(0)
scope = USBScope(addr='USB0::0x1AB1::0x0514::DS7F222900085::INSTR')
specAn = USBSpectrumAnalyzer(addr="USB0::6833::2400::DSA8A223200862::0::INSTR")
# specAn = USBSpectrumAnalyzer()
piezo = PiezoTIM101('65863991')
pxpitch = 5.5e-6
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
kx = np.fft.fftfreq(w, pxpitch)*1e-6
ky = np.fft.fftfreq(h, pxpitch)*1e-6
setup = Homodyne(piezo, specAn, scope, cam)
# positions, k_mirror = setup.calib_fringes(start=100, stop=60000, steps=100, plot=True)
# setup.move_to_k(0.005e6)
cont = input("Electronic noise calibration, block diodes and press any key" +
             " to continue ")
if cont is not None:
    elec, time_e, lo_elec, time_lo_elec = setup.measure_once()
else:
    sys.exit()
# cont = input("Electronic noise calibration recorded, press any key to" +
#              " continue ")
# cont = input("Vacuum calibration, block signal and press any key" +
#              " to continue ")
# if cont is not None:
#     vacuum, time_v, lo_vacuum, time_lo_vacuum = setup.specAn.zero_span()
# else:
#     sys.exit()
# cont = input("Vacuum calibration recorded, press any key to" +
#              " continue ")
# if cont is not None:
#     spectra, time, k_actual = setup.scan_k()
# else:
#     sys.exit()
piezo.disconnect()
cam.release()
specAn.close()
scope.close()
print('All instruments disconnected ! Goodbye !')
elec = np.asarray(elec)
indices = np.linspace(0, len(elec)-1, len(elec), dtype=int)
lo_elec = np.asarray(lo_elec)
# vacuum = np.asarray(vacuum)
# spectra = np.asarray(spectra)
elec_in_V = np.exp(elec/10)
lo_elec_in_V = np.exp(lo_elec[0,indices]/10)
plt.plot(time_e, elec_in_V)
plt.plot(time_e, lo_elec_in_V)
# vacuum_in_V = np.exp(vacuum/10)
# spectra_in_V = np.exp(spectra/10) - vacuum_in_V - elec_in_V
# spec_k = np.mean(spectra_in_V[0, :, :], axis=1)
# std = np.std(spectra_in_V[0, :, :], axis=1)
# fig, ax = plt.subplots(1, 1)
# fig.suptitle("Noise spectrum vs LO angle")
# ax.set_xlabel("Wavevector in $\\mu m^{-1}$")
# ax.set_ylabel("Signal in mV")
# ax.plot(k_actual*1e-6, np.mean(vacuum_in_V)*np.ones(k_actual.shape))
# ax.errorbar(k_actual*1e-6, spec_k*1e3, std)
plt.legend(["Noise", "LO"])
plt.yscale('log')
plt.show()
