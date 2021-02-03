# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:01:54 2021

@author: Tangui ALADJIDI
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
import EasyPySpin
import cv2
import sys
import os

plt.switch_backend('Qt5Agg')
plt.ion()

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

class Homodyne:

    def __init__(self, piezo: PiezoTIM101, specAn: USBSpectrumAnalyzer,
                 scope: USBScope, cam: EasyPySpin.VideoCapture):
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

    def calib_fringes(self, channel: int = 1, start: int = 100,
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
        Kx, Ky = np.meshgrid(kx, ky)
        K = np.fft.fftshift(np.sqrt(Kx**2 + Ky**2))
        roi = K>6e2 #hardcoded, not great
        ret = False
        tries = 0
        while ret == False and tries < 10:
            ret, frame_calib = self.cam.read()
            tries += 1
        if ret == False:
            print("ERROR : Could not grab frame")
            sys.exit()
        fig0, ax = plt.subplots(1, 1)
        ax.imshow(frame_calib, cmap="gray")
        ax.set_title("Calibration picture : should be 0 fringes")
        plt.show(block=False)
        pos_range = np.linspace(start, stop, steps)
        positions = np.empty(pos_range.shape)
        k_mirror = np.empty(len(pos_range))
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=False, sharey=False)
        ax0.set_title("Image")
        ax1.set_title("Fourier transform")
        im0 = ax0.imshow(np.ones((frames.shape[0], frames.shape[1])),
                    cmap="gray", vmin=0, vmax=255)
        im1 = ax1.imshow(np.ones((frames.shape[0], frames.shape[1])),
                         extent=[np.min(kx)*1e-6, np.max(kx)*1e-6,
                                 np.min(ky)*1e-6, np.max(ky)*1e-6],
                         vmin=4,
                         vmax=17)
        ax1.set_xlabel("$k_x$ in $\\mu m^{-1}$")
        ax1.set_ylabel("$k_y$ in $\\mu m^{-1}$")
        # jogs along the specified range
        for counter, pos in enumerate(pos_range):
            positions[counter] = self.piezo.move_to(channel, pos)
            ret = False
            tries = 0
            while ret == False and tries < 10:
                ret, frame = self.cam.read()
                tries += 1
            if ret == False:
                print("ERROR : Could not grab frame")
                sys.exit()
            frame_fft = np.fft.fftshift(np.fft.fft2(frame))
            roi = (K>6e2)
            im_filt = frame_fft*roi
            max = np.where(im_filt == np.max(im_filt))
            k_mirror[counter] = K[max][0]
            if (counter+1)%10==0:
                im0.set_data(frame)
                im1.set_data(np.log(np.abs(frame_fft)))
                circle = Circle(max, radius = 5, fill = False)
                ax1.add_patch(circle)
                im1.set_clip_path(circle)
                fig.suptitle(f"Grabbed frame {counter+1}/{len(pos_range)}")
                mypause(0.5)
                fig.canvas.draw()
        plt.show(block=False)
        # returns piezo to original position
        self.piezo.move_to(channel, 0)
        # captures image to check the angle
        ret = False
        counter = 0
        while not(ret) and counter < 10:
            ret, frame_return = self.cam.read()
            counter += 1
        if not(ret):
            print("ERROR : Could not grab frame")
            sys.exit()
        # ax.imshow(frame_return, cmap="gray")
        # ax.set_title("Back to the start : should be 0 fringes")
        # plt.show()
        return positions, k_mirror
