# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:01:54 2021
@author: Tangui ALADJIDI
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
import EasyPySpin
import cv2
import sys
import os
import configparser

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
        conf = configparser.ConfigParser()
        conf.read("homodyne.conf")
        self.pos_to_k = [float(conf["Calib"][f"pos_to_k{n+1}"]) for n in range(4)]

    def calib_fringes(self, channel: int = 1, start: int = 100,
                      stop: int = 10000, steps: int = 200,
                      pxpitch: float = 5.5e-6, plot=False) -> np.ndarray:
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
        roi = K > 6e2  # hardcoded, not great
        ret = False
        tries = 0
        while ret is False and tries < 10:
            ret, frame_calib = self.cam.read()
            tries += 1
        if ret is False:
            print("ERROR : Could not grab frame")
            sys.exit()
        if plot:
            fig0, ax = plt.subplots(1, 1)
            ax.imshow(frame_calib, cmap="gray")
            ax.set_title("Calibration picture : should be 0 fringes")
            plt.show(block=False)
        pos_range = np.linspace(start, stop, steps)
        positions = np.empty(pos_range.shape)
        k_mirror = np.empty(len(pos_range))
        if plot:
            fig, (ax0, ax1) = plt.subplots(1, 2, sharex=False, sharey=False)
            ax0.set_title("Image")
            ax1.set_title("Fourier transform")
            im0 = ax0.imshow(np.ones((frame_calib.shape[0],
                                      frame_calib.shape[1])),
                             cmap="gray", vmin=0, vmax=255)
            im1 = ax1.imshow(np.ones((frame_calib.shape[0],
                                      frame_calib.shape[1])),
                             extent=[np.min(kx)*1e-6, np.max(kx)*1e-6,
                                     np.min(ky)*1e-6, np.max(ky)*1e-6],
                             vmin=6, vmax=14)
            ax1.set_xlabel("$k_x$ in $\\mu m^{-1}$")
            ax1.set_ylabel("$k_y$ in $\\mu m^{-1}$")
        # jogs along the specified range
        for counter, pos in enumerate(pos_range):
            positions[counter] = self.piezo.move_to(channel, pos)
            ret = False
            tries = 0
            while ret is False and tries < 10:
                ret, frame = self.cam.read()
                tries += 1
            if ret is False:
                print("ERROR : Could not grab frame")
                sys.exit()
            frame_fft = np.fft.fftshift(np.fft.fft2(frame))
            roi = K > 6e2
            im_filt = np.abs(frame_fft)*roi
            max = np.where(im_filt == np.max(im_filt))
            k_mirror[counter] = K[max][0]
            if (counter+1) % 10 == 0 and plot:
                im0.set_data(frame)
                im1.set_data(np.log(np.abs(frame_fft)))
                fig.suptitle(f"Grabbed frame {counter+1}/{len(pos_range)}")
                mypause(0.5)
                fig.canvas.draw()
        if plot:
            plt.show(block=False)
        # returns piezo to original position
        pos = self.piezo.move_to(channel, 0)
        if pos != 0:
            pos = self.piezo.move_to(channel, 0)

        # captures image to check the angle
        ret = False
        counter = 0
        while not(ret) and counter < 10:
            ret, frame_return = self.cam.read()
            counter += 1
        if not(ret):
            print("ERROR : Could not grab frame")
            sys.exit()
        # record calibration
        # linear fit
        a, b = np.polyfit(positions, k_mirror, 1)
        # write to file for future use
        conf = configparser.ConfigParser()
        conf.read("homodyne.conf")
        conf["Calib"][f"pos_to_k{channel}"] = str(a)
        with open('homodyne.conf', 'w') as configfile:
            conf.write(configfile)
        self.pos_to_k[channel-1] = a
        fig1, ax = plt.subplots(1, 1, figsize=(12, 20))
        ax.plot(positions, k_mirror*1e-6)
        ax.plot(positions, 1e-6*(a*positions+b), color='red', linestyle='--')
        textstr = f"a = {np.round(a, decimals=3)*1e-6} " + \
                  "$\\mu m^{-1}$/step" + \
                  f"\n b = {np.round(b, decimals=3)*1e-6} "+"$\\mu m^{-1}$"
        props = dict(boxstyle='square', facecolor='grey', alpha=0.5)
        ax.text(0.90, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.set_title("Mirror angle calibration")
        ax.set_xlabel("Piezo position in steps")
        ax.set_ylabel("Wavenumber of the LO in $\\mu m^{-1}$")
        ax.legend(["Experimental", "Linear fit"])
        plt.savefig("calibration.png")
        plt.show(block=False)
        return positions, k_mirror

    def move_to_k(self, k: float, channels: list = [1]):
        """
        Moves the LO to a specified k value
        :param k: Wavevector in m^{-1}
        :type k: float
        :param channels: Channels to move, defaults to [1]
        :type channels: list, optional
        :return: None
        :rtype: None
        """
        # check that the channels list provided is correct
        if len(channels) > 4:
            print("ERROR : Invalid channel list provided" +
                  " (List too long)")
            sys.exit()
        for chan in channels:
            if chan > 4:
                print("ERROR : Invalid channel list provided" +
                      " (Channels are 1,2,3,4)")
                sys.exit()
        for nbr, chan in enumerate(channels):
            pos_to_k = self.pos_to_k[nbr]
            pos_tgt = int(k/pos_to_k)
            self.piezo.move_to(chan, pos_tgt)

    def scan_k(self, k0: float = 0, k1: float = 0.005e6, steps: int = 50,
               channels: list = [1], rbw: float = 100e3, vbw: float = 30,
               swt: float = 50e-3, trig: bool = False) -> np.ndarray:
        """
        Scans the given k values and takes a spectrum for each k value
        :param k0: Start wavevector in m^{-1}, defaults to 0
        :type k0: float, optional
        :param k1: Stop wavevector in m^{-1}, defaults to 0.001e6
        :type k1: float, optional
        :param steps: Number of steps, defaults to 50
        :type steps: int, optional
        :param channels: channels list, defaults to [1]
        :type channels: list, optional
        :param float rbw: Resolution bandwidth
        :param float vbw: Video bandwidth
        :param float swt: Total measurement time
        :param bool trig: External trigger
        :return: data, time for data and time
        :return: Array of spectra
        :rtype: np.ndarray
        """
        # check that the channels list provided is correct
        if len(channels) > 4:
            print("ERROR : Invalid channel list provided" +
                  " (List too long)")
            sys.exit()
        for chan in channels:
            if chan > 4:
                print("ERROR : Invalid channel list provided" +
                      " (Channels are 1,2,3,4)")
                sys.exit()
        # puts the specAn in zero span mode and retrieve a spectrum to get the
        # data formats
        k_values = np.linspace(k0, k1, steps)
        data, time = self.specAn.zero_span()
        spectra = np.empty((len(channels), len(k_values), len(time)))
        for nbr, chan in enumerate(channels):
            for counter_k, k in enumerate(k_values):
                data, time = self.specAn.zero_span()
                spectra[nbr, counter_k, :] = data
        return spectra, time
