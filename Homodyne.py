# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:01:54 2021
@author: Tangui ALADJIDI
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101
import EasyPySpin
import cv2
import sys
import configparser
import traceback
import scipy.fft as fft
import pyfftw
import multiprocessing

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
        self.pos_to_k = [float(conf["Calib"][f"pos_to_k{n+1}"]) for n
                         in range(4)]

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

    def move_to_k(self, k: float, channels: list = [1]) -> float:
        """
        Moves the LO to a specified k value
        :param k: Wavevector in m^{-1}
        :type k: float
        :param channels: Channels to move, defaults to [1]
        :type channels: list, optional
        :return: k_actual the actual angle reached
        :rtype: float
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
            pos = self.piezo.move_to(chan, pos_tgt)
            k_actual = pos*pos_to_k
            prt = np.round(k_actual*1e-6, decimals=6)
            sys.stdout.write(f", k = {prt} um^-1")
            return k_actual

    def __measure_specAn(self, output: list,
                         center: float = 1e6, rbw: float = 100e3,
                         vbw: float = 30, swt: float = 50e-3,
                         trig: bool = False) -> np.ndarray:
        """
        Private copy of a SpecAn zero span measure with output list to
        stock the output of the measurement for parallel execution
        :param list output: Description of parameter `output`.
        :return: None
        """
        data, time = self.specAn.zero_span(center, rbw, vbw, swt, trig)
        output.append((data, time))

    def __measure_scope(self, output: list, channels: list = [1]):
        """
        Private copy of a get_waveform Scope measure with output list to
        stock the output of the measurement for parallel execution
        :param list output: Description of parameter `output`.
        :return: None
        """
        data, time = self.scope.get_waveform(channels)
        output.append((data, time))

    def measure_once(self, channels: list = [1], center: float = 1e6,
                     rbw: float = 100e3, vbw: float = 30, swt: float = 50e-3,
                     trig: bool = False) -> np.ndarray:
        """
        Does a single noise measurement
        :param channels: channels list, defaults to [1]
        :type channels: list, optional
        :param float center: Center frequency in Hz
        :param float rbw: Resolution bandwidth
        :param float vbw: Video bandwidth
        :param float swt: Total measurement time
        :param bool trig: External trigger
        :return: data, time for data and time
        :return: Tuple of spectrum / LO power ((channels, time), (LO, time_LO))
        :rtype: np.ndarray
        """

        output_specAn = []
        output_scope = []
        # measure simultaneously on the SpecAnalyzer as well as on the Scope
        p0 = multiprocessing.Process(target=self.__measure_specAn,
                                     args=(output_specAn, center, rbw, vbw,
                                           swt, trig))
        p1 = multiprocessing.Process(target=self.__measure_scope,
                                     args=(output_scope, channels))
        p0.start()
        p1.start()
        p0.join()
        p1.join()
        data_scope = output_scope[0]
        data_specAn = output_specAn[0]
        time_scope = output_scope[1]
        time_specAn = output_specAn[1]
        del output_scope, output_specAn
        return data_specAn, time_specAn, data_scope, time_scope

    def scan_k(self, k0: float = 10, k1: float = 0.01e6, steps: int = 50,
               channels: list = [1], center: float = 1e6, rbw: float = 100e3,
               vbw: float = 30, swt: float = 50e-3,
               trig: bool = False) -> np.ndarray:
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
        :param float center: Center frequency in Hz
        :param float rbw: Resolution bandwidth
        :param float vbw: Video bandwidth
        :param float swt: Total measurement time
        :param bool trig: External trigger
        :return: data, time for data and time
        :return: Array of spectra (channels, k_values, time)
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
        k_actual = np.empty(k_values.shape)
        data, time = self.specAn.zero_span(center, rbw, vbw, swt, trig)
        spectra = np.empty((len(channels), len(k_values), len(time)))
        for nbr, chan in enumerate(channels):
            for counter_k, k in enumerate(k_values):
                try:
                    k_actual[counter_k] = self.move_to_k(k, channels=[chan])
                    data, time = self.specAn.zero_span(center, rbw, vbw, swt,
                                                       trig)
                    spectra[nbr, counter_k, :] = data
                except Exception:
                    print("ERROR : Could not move")
                    print(traceback.format_exc())
            self.piezo.move_to(chan, 0)
        return spectra, time, k_actual

    def __get_visibility(frame, fft_side=None, fft_center=None,
                         fft_obj_s=None, ifft_obj_s=None, ifft_obj_c=None):
        """Gets the visibility of a given fringe pattern using Fourier filtering

        :param np.ndarray frame: Fringe pattern.
        :param pyfftw.empty_aligned fft_side: Array to perform fft on for side
            peak.
        :param pyfftw.empty_aligned fft_center: Array to perform fft on for
            center peak.
        :param pyfftw.FFTW fft_obj_s: Fft instance for the side peak
        :param pyfftw.FFTW ifft_obj_s: Fft instance for the side peak
        :param pyfftw.FFTW ifft_obj_c: Fft instance for the center peak
        :return: Max of visibility and visibility map
        :rtype: (float, np.ndarray[np.float32, ndim=2])

        """
        def shift5(arr, numi, numj, fill_value=0):
            """Fast array shifting

            :param np.ndarray arr: Array to shift
            :param int numi: Pixel to shift for row number
            :param int numj: Pixel to shift for column number
            :param fill_value: Filling value
            :return: The shifted array
            :rtype: depends on fill value type np.ndarray[np.float32, ndim=2]

            """
            result = np.empty_like(arr)
            if numi > 0:
                result[:numi, :] = fill_value
                result[numi:, :] = arr[:-numi, :]
            elif numi < 0:
                result[numi:, :] = fill_value
                result[:numi, :] = arr[-numi:, :]
            if numj > 0:
                result[:, :numj] = fill_value
                result[:, numj:] = arr[:, :-numj]
            elif numj < 0:
                result[:, numj:] = fill_value
                result[:, :numj] = arr[:, -numj:]
            else:
                result[:] = arr
            return result

        if fft_side is not None and fft_center is not None:
            fft_side[:] = frame
            fft_center[:] = frame
        else:
            fft_side = np.copy(frame)
            fft_center = np.copy(frame)
        del frame
        kx = fft.fftshift(np.fft.fftfreq(fft_side.shape[1], 5.5e-6))
        ky = fft.fftshift(np.fft.fftfreq(fft_side.shape[0], 5.5e-6))
        Kx, Ky = np.meshgrid(kx, ky)
        K = np.sqrt(Kx**2 + Ky**2)
        roi = np.zeros(K.shape, dtype=np.complex64)
        roic = np.zeros(K.shape, dtype=np.complex64)
        roi[Kx > 10e2] = 1
        roic[K <= 10e2] = 1
        if fft_obj_s is not None:
            fft_side = fft.fftshift(fft_obj_s(fft.fftshift(fft_side)))
        else:
            fft_side = fft.fftshift(fft.fft2(fft.fftshift(fft_side)))
        fft_center[:] = fft_side*roic
        fft_side[:] = fft_side*roi
        max = np.where(np.abs(fft_side) == np.max(np.abs(fft_side)))
        fft_side = shift5(fft_side, fft_side.shape[0]//2-max[0][0],
                          fft_side.shape[1]//2-max[1][0])
        if ifft_obj_s is not None:
            vis = fft.fftshift(ifft_obj_s(fft.ifftshift(fft_side)))
        else:
            vis = fft.fftshift(fft.ifft2(fft.ifftshift(fft_side)))
        if ifft_obj_c is not None:
            fft_center = fft.fftshift(ifft_obj_c(fft.ifftshift(fft_center)))
        else:
            fft_center = fft.fftshift(fft.ifft2(fft.ifftshift(fft_center)))
        vis /= fft_center
        # filter bad pixels
        vis[:10, :] = 0
        vis[:, :10] = 0
        vis[-10:, :] = 0
        vis[:, -10:] = 0
        return np.max(np.abs(vis[np.abs(vis) > 0.1])), np.abs(vis)

    def monitor_visibility(self):
        """Monitor fringe visibility with a live view of the camera. Will
        display a window with a preview of the camera, the visibility and
        dynamic graph of the maximum visibility

        :return: None
        :rtype: Nonetype

        """
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[0, 1]))
        # spans two rows:
        ax.append(fig.add_subplot(gs[1, :]))
        xs = list(range(0, 200))
        ys = [0] * len(xs)
        ax[2].set_ylim((0, 100))
        im = ax[0].imshow(np.zeros((w, h)), vmin=0, vmax=255, cmap='gray')
        im1 = ax[1].imshow(np.zeros((w, h)), vmin=0, vmax=1)
        line, = ax[2].plot(xs, ys)
        cbar = fig.colorbar(im, ax=ax[0])
        cbar1 = fig.colorbar(im1, ax=ax[1])
        cbar.set_label("Intensity", rotation=270)
        cbar1.set_label("Visibility", rotation=270)

        ax[0].set_title("Camera")
        ax[1].set_title("Fringe visibility")
        ax[2].set_title("Fringe visibility")
        ax[2].set_xlabel("Samples")
        ax[2].set_ylabel("Visibility in %")
        plt.tight_layout()
        fft_side = pyfftw.empty_aligned((w, h), dtype=np.complex64)
        fft_center = pyfftw.empty_aligned((w, h), dtype=np.complex64)
        fft_obj_s = pyfftw.builders.fft2(fft_side,
                                         overwrite_input=True,
                                         threads=multiprocessing.cpu_count(),
                                         planner_effort="FFTW_MEASURE")
        ifft_obj_s = pyfftw.builders.ifft2(fft_side,
                                           overwrite_input=True,
                                           threads=multiprocessing.cpu_count(),
                                           planner_effort="FFTW_MEASURE")
        ifft_obj_c = pyfftw.builders.ifft2(fft_center,
                                           overwrite_input=True,
                                           threads=multiprocessing.cpu_count(),
                                           planner_effort="FFTW_MEASURE")

        def animate(i, ys, fft_side, fft_center, fft_obj_s, ifft_obj_s,
                    ifft_obj_c):
            ret, frame = self.cam.read()
            vis, Vis = self.__get_visibility(frame, fft_side, fft_center,
                                             fft_obj_s, ifft_obj_s, ifft_obj_c)
            im.set_data(frame)
            im1.set_data(Vis)
            # Add y to list
            ys.append(100*vis)

            # Limit y list to set number of items
            ys = ys[-len(xs):]

            # Update line with new Y values
            line.set_ydata(ys)

            return im, im1, line,

        FuncAnimation(fig, animate,
                      fargs=(ys, fft_side, fft_center, fft_obj_s, ifft_obj_s,
                             ifft_obj_c),
                      interval=50, blit=True)
        plt.show()
