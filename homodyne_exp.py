import numpy as np
import cv2
import EasyPySpin
import matplotlib.pyplot as plt
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
from Homodyne import Homodyne

piezo = PiezoTIM101('65863991')
piezo.zero()
piezo.move(channel=1, pos=100)
piezo.disconnect()

piezo = PiezoTIM101('65863991')
cam = EasyPySpin.VideoCapture(0)
scope = USBScope()
specAn = USBSpectrumAnalyzer(addr="USB0::6833::2400::DSA8A223200862::0::INSTR")
setup = Homodyne(piezo, specAn, scope, cam)
setup.calib_fringes()
