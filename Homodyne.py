# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:01:54 2021

@author: Tangui ALADJIDI
"""

import numpy as np
import matplotlib.pyplot as plt
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101


class Homodyne:

    def __init__(self, piezo: PiezoTIM101, specAn: USBSpectrumAnalyzer,
                 scope: USBScope, cam):
        """Instantiates the homodyne setup with a piezo to move the LO angle,
        a spectrum analyzer, an oscilloscope and a camera.

        :param PiezoTIM101 piezo: Piezo to jog
        :param USBSpectrumAnalyzer specAn: spectrum analyzer
        :param USBScope scope: oscilloscope
        :param EasyPySpin cam: camera, EasyPySpin instance
        :return: Homodyne object
        :rtype: Homodyne

        """

        self.piezo = piezo
        self.specAn = specAn
        self.scope = scope
        self.cam = cam

    def calib_fringes(self, channel: int = 1, start: int = 0,
                      stop: int = 10000) -> np.ndarray:
        """Function to calibrate the k values accessed by the local oscillator

        :param PiezoTIM101 piezo: piezo to actuate
        :param int channel: Piezo channel to actuate
        :param int start: Start position of the piezo
        :param int stop: Stop position of the piezo
        :return: Description of returned object.
        :rtype: np.ndarray

        """
        ret = False
        counter = 0
        while not(ret) and counter < 10:
            ret, frame_calib = self.cam.read()
            counter += 1
        plt.imshow(frame_calib, cmap="gray")
        plt.title("Calibration picture : should be 0 fringes")
        plt.show()
