# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:01:54 2021

@author: Tangui ALADJIDI
"""

import numpy as np
import matplotlib.pyplot as plt
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
import cv2


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
                      stop: int = 10000, steps: int = 200) -> np.ndarray:
        """Function to calibrate the k values accessed by the local oscillator

        :param PiezoTIM101 piezo: piezo to actuate
        :param int channel: Piezo channel to actuate
        :param int start: Start position of the piezo
        :param int stop: Stop position of the piezo
        :return: Description of returned object.
        :rtype: np.ndarray

        """
        # Gets the camera size
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        ret = False
        counter = 0
        while not(ret) and counter < 10:
            ret, frame_calib = self.cam.read()
            counter += 1
        if not(ret):
            print("ERROR : Could not grab frame")
            sys.exit()
        plt.imshow(frame_calib, cmap="gray")
        plt.title("Calibration picture : should be 0 fringes")
        plt.show(block=False)
        pos_range = np.linspace(start, stop, steps)
        positions = np.empty(pos_range.shape)
        frames = np.empty((h, w, len(pos_range))
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        im = ax.imshow(np.ones((frames.shape[0], frames.shape[1]), cmap="gray")
        # jogs along the specified range
        for counter, pos in enumerate(pos_range):
            positions[counter] = self.piezo.move_to(channel, pos)
            ret = False
            counter = 0
            while not(ret) and counter < 10:
                ret, frames[:, :, counter] = self.cam.read()
                counter += 1
            if not(ret):
                print("ERROR : Could not grab frame")
                sys.exit()
            im.set_data(frames[:, :, counter])
            ax.set_title(f"Grabbed frame {counter+1}/{len(pos_range)}")
            fig.canvas.draw()
        plt.show(block=False)
        # returns piezo to original position
        self.piezo.move_to(channel, start)
        # captures image to check the angle
        ret = False
        counter = 0
        while not(ret) and counter < 10:
            ret, frame_return = self.cam.read()
            counter += 1
        if not(ret):
            print("ERROR : Could not grab frame")
            sys.exit()
        plt.imshow(frame_return, cmap="gray")
        plt.title("Back to the start : should be 0 fringes")
        plt.show(block=False)
