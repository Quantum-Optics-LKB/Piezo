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
                      stop: int = 10000, steps: int = 200,
                      pxpitch: float = 5.5e-6) -> np.ndarray:
        """Function to calibrate the k values accessed by the local oscillator

        :param PiezoTIM101 piezo: piezo to actuate
        :param int channel: Piezo channel to actuate
        :param int start: Start position of the piezo
        :param int stop: Stop position of the piezo
        :param float pxpitch: Pixel pitch of the camera
        :return: Description of returned object.
        :rtype: np.ndarray

        """
        # Gets the camera size
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        kx = np.fft.fftfreq(w, pxpitch)
        ky = np.fft.fftfreq(h, pxpitch)
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
        frames_fft = np.empty((h, w, len(pos_range))
        fig = plt.figure(0)
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        ax0.set_title("Image")
        ax1.set_title("Fourier transform")
        im0 = ax.imshow(np.ones((frames.shape[0], frames.shape[1]),
                        cmap="gray")
        im1 = ax.imshow(np.ones((frames.shape[0], frames.shape[1]))
        ax1.set_xlabel("$k_x$ in $\\mu m^{-1}$")
        ax1.set_ylabel("$k_y$ in $\\mu m^{-1}$")
        ax1.set_xticks(kx*1e-6)
        ax1.set_yticks(ky*1e-6)
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
            frames_fft[:, :, counter] = np.fft.fftshift(
                                            np.fft.fft2(frames[:, :, counter]))
            im0.set_data(frames[:, :, counter])
            im1.set_data(frames_fft[:, :, counter])
            fig.suptitle(f"Grabbed frame {counter+1}/{len(pos_range)}")
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
        return frames, frames_fft
