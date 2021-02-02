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
