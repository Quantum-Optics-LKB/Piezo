import numpy as np
import cv2
import EasyPySpin
import matplotlib.pyplot as plt
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
from Homodyne import Homodyne
import os

plt.switch_backend('Qt5Agg')
plt.ion()

# piezo = PiezoTIM101('65863991')
# piezo.move_to(channel=1, pos=-100)
# piezo.move_to(channel=1, pos=0)
# piezo.disconnect()

cam = EasyPySpin.VideoCapture(0)
scope = None
specAn = USBSpectrumAnalyzer(addr="USB0::6833::2400::DSA8A223200862::0::INSTR")
piezo = PiezoTIM101('65863991')
pxpitch = 5.5e-6 
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
kx = np.fft.fftfreq(w, pxpitch)*1e-6
ky = np.fft.fftfreq(h, pxpitch)*1e-6
setup = Homodyne(piezo, specAn, scope, cam)
frames, positions = setup.calib_fringes(start=100, stop=10000, steps=200)

piezo.disconnect()
cam.release()
specAn.close()
