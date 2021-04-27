import numpy as np
import cv2
import EasyPySpin
import matplotlib.pyplot as plt
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
from Homodyne import Homodyne
import sys
import time_s

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
# print("Capturing a frame to measure probe's k ...")
# k_probe = setup.get_k_from_frame()
T_cell = setup.get_cell_temp()
# k_probe = 0.009868789407128323*1e6
# # print(f"Done ! Probe k = {k_probe*1e-6} um^-1")
# cont = input("Electronic noise calibration, block diodes and press any key" +
#              " to continue ")
# if cont is not None:
#     elec, time_e = setup.specAn.zero_span(rbw=300e3, vbw=300)
# else:
#     sys.exit()
# cont = input("Electronic noise calibration recorded, press any key to" +
#               " continue ")
# cont = input("Vacuum calibration, block signal and press any key" +
#               " to continue ")
# if cont is not None:
#     vacuum, time_v, lo_vacuum, time_lo_vacuum = setup.measure_once(rbw=300e3, vbw=300)
# else:
#     sys.exit()
# cont = input("Vacuum calibration recorded unblock signal and press any key" +
#              " to continue ")
# if cont is not None:
#     # spectra, time_s, lo, time_scope, k_actual = setup.scan_k(k0=-10,
#     #                                                        k1=-0.02e6,
#     #                                                        steps=150,
#     #                                                        rbw=300e3, vbw=300)
#     spectra, time_s, lo, time_scope = setup.scan_f(rbw=300e3, vbw=300)
# else:
#     sys.exit()
piezo.disconnect()
cam.release()
specAn.close()
scope.close()
print('\n All instruments disconnected ! Goodbye !')
# elec = np.asarray(elec)
# indices = np.linspace(0, len(elec)-1, len(elec), dtype=int)
# lo_vacuum = np.asarray(lo_vacuum)
# vacuum = np.asarray(vacuum)
# spectra = np.asarray(spectra)
# lo = np.asarray(lo)
# elec_in_V = np.exp(elec/10)
# vacuum_in_V = np.exp(vacuum/10) - elec_in_V
# lo_vacuum_in_V = np.exp(lo_vacuum[indices]/10)
# spectra_in_V = np.exp(spectra/10) - elec_in_V
# # lo_in_V = np.exp(lo[:, :, indices]/10) - elec_in_V
# lo_in_V = np.exp(lo[:, indices]/10) - elec_in_V
#
# # plt.plot(time_e, elec_in_V)
# # plt.plot(time_e, vacuum_in_V)
# # plt.plot(time_e, lo_vacuum_in_V)
# # plt.plot(time_s, spectra_in_V[0, 27, :])
# plt.plot(time_s, spectra_in_V[0, :])
# freqs = np.linspace(1e6, 45e6, 50)
# # plt.plot(time_s, lo_in_V[0, 0, :])
# # plt.legend(["Spectrum", "LO"])
# # plt.title(f"Spectrum at k = {np.round(k_actual[27]*1e-6, decimals=6)}" +
# #           " $\\mu m^{-1}$")
# plt.title(f"Spectrum at f = {freqs[0]*1e-6} MHz")
# plt.xlabel("Time in s")
# plt.ylabel("Signal in V")
# plt.show()
# # spec_k = np.mean(spectra_in_V[0, :, :], axis=1)
# # lo_k = np.mean(lo_in_V[0, :, :], axis=1)
# # std = np.std(spectra_in_V[0, :, :], axis=1)
# # fig, ax = plt.subplots(1, 2)
# # fig.suptitle("Noise spectrum vs LO angle")
# # ax[0].set_xlabel("Wavevector in $\\mu m^{-1}$")
# # ax[0].set_ylabel("Signal in mV")
# # ax[1].set_xlabel("Wavevector in $\\mu m^{-1}$")
# # ax[1].set_ylabel("Signal in V")
# # ax[0].plot(k_actual*1e-6, np.mean(vacuum_in_V)*1e3*np.ones(k_actual.shape))
# # ax[0].errorbar(k_actual*1e-6, spec_k*1e3, std)
# # ax[0].axvline(x=-k_probe*1e-6, linestyle='dashed')
# # ax[0].legend(["Vacuum", "Probe k", "Signal"])
# # ax[0].set_yscale('log')
# # ax[0].set_title("Signal power")
# # ax[1].plot(k_actual*1e-6, lo_k)
# # ax[1].set_title("CH1 power in V")
# # plt.show()
#
# spec_f = np.mean(spectra_in_V, axis=1)
# lo_f = np.mean(lo_in_V, axis=1)
# std = np.std(spectra_in_V, axis=1)
# fig, ax = plt.subplots(1, 2)
# fig.suptitle("Noise spectrum vs frequency")
# ax[0].set_xlabel("Zero span center frequency in MHz")
# ax[0].set_ylabel("Signal in mV")
# ax[1].set_xlabel("Zero span center frequency in MHz")
# ax[1].set_ylabel("Signal in V")
# ax[0].plot(freqs*1e-6, np.mean(vacuum_in_V)*1e3*np.ones(freqs.shape))
# ax[0].errorbar(freqs*1e-6, spec_f*1e3, std)
# ax[0].legend(["Vacuum", "Signal"])
# ax[0].set_yscale('log')
# ax[0].set_title("Signal power")
# ax[1].plot(freqs*1e-6, lo_f)
# ax[1].set_title("CH1 power in V")
# plt.show()
