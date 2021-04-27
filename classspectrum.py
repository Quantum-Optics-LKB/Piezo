# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:38:28 2021

@author: Thomas Picot / Tangui Aladjidi
"""

import numpy as np
import scipy.constants as cst
from scipy.special import erfc
from scipy.special import wofz

# voigt function
#def V(z):
#    return 1j * np.exp(z ** 2) * erfc(z)

def V(z):
    return 1j*wofz(1j*z)


"""
don't touch it
"""


def Ndensity(temp, omega,deg):
    N = (133.323 * 10 ** (15.88253 - (4529.635 / temp) + 0.00058663 * temp - 2.99138 * np.log10(temp))) / (
                cst.Boltzmann * temp)
    return N/deg


# change the temperature parameter in the function
def varsigma(temp):
    sigma = 2 * np.pi * np.sqrt((2 * cst.k * temp) / (1.4099931997e-25 * 7.807864080702083e-7 ** 2))
    return sigma


class DisplaySpectrum:
    def __init__(self, gamma=0.5 * 3.81138e7, k=2*np.pi/7.807864080702083e-7, v=2e3, m=1.4099931997e-25):
        self.h = cst.hbar
        self.eps_zero = cst.epsilon_0
        self.gamma = gamma
        self.k = k
        self.d = 3.24627e-29
        self.v = v
        self.m = m

        # Rb 87

        # F_g = 2 --> F_e = 1, 2, 3
        self.w_i = -2735.05e6
        self.w_ii = -2578.11e6
        self.w_iii = -2311.26e6

        # F_g = 1 --> F_e = 0, 1, 2
        self.w_ib = 4027.403e6
        self.w_iib = 4099.625e6
        self.w_iiib = 4256.57e6

        # Rb 85

        # F_g = 3 --> F_e = 2, 3, 4
        self.w_j = -1371.29e6
        self.w_jj = -1307.87e6
        self.w_jjj = -1186.91e6

        # F_g = 2 --> F_e = 1, 2, 3
        self.w_jb = 1635.454e6
        self.w_jjb = 1664.714e6
        self.w_jjjb = 1728.134e6

    def varomega(self):
        omega = np.linspace(-6e9, 6e9, 3000)
        #omega = 2*np.pi*np.linspace(-5e9, 5e9, 3000)     # useless to touch it
        return omega

    def fracRb(self, frac, temp, omega,deg):
        """
        DESCRIPTION:
        PARAMS:         frac-> percent of Rb85/Rb87
                        temp-> temperature of the cell [K]
                        deg-> degeneracy of the ground state according to the isotope : 8 for 87 and 12 for 85
        RETURNS:        Float
        """
        N = frac*Ndensity(temp, omega,deg)
        return N

    def imaginaryPartOfSusceptibility(self, C_f, frac, temp,omega, detuning, deg):
        """
        DESCRIPTION:    I think the desc is in function.
        PARAMS:         C_f->C_f^2 is the transition coefficient
                        frac-> percent of Rb85/Rb87
                        temp-> temperature of the cell [K]
                        omega-> frequency of light
                        detuning-> transition frequency
                        deg-> degeneracy of the ground state according to the isotope : 8 for 87 and 12 for 85
        RETURNS:        1darray
        """

        N = self.fracRb(frac, temp, omega,deg)
        delta = 2*np.pi*(omega - detuning)
        voigt_arg = (self.gamma - 1j * delta) / varsigma(temp)
        absorption = C_f*(N * (self.d ** 2) * np.sqrt(np.pi) / (self.h * self.eps_zero *varsigma(temp))) * V(voigt_arg).imag
        #print(V(self.gamma, varsigma(temp)/np.sqrt(2) ,-1*delta).imag)
        return absorption

    def alpha(self, C_f, frac, temp,omega, detuning, deg):
        """
        DESCRIPTION:    absorption coefficient
        PARAMS:         C_f->C_f^2 is the transition coefficient
                        frac-> percent of Rb85/Rb87
                        temp-> temperature of the cell [K]
                        omega-> frequency of light
                        detuning-> transition frequency
                        deg-> degeneracy of the ground state according to the isotope : 8 for 87 and 12 for 85
        RETURNS:        1darray
        """

        alpha = self.k*self.imaginaryPartOfSusceptibility(C_f, frac, temp, omega,omega, detuning, deg)
        return alpha

    def transmission(self, frac, temp, long, omega):
        """
        DESCRIPTION:    Function that contains data of the theoretical transmission of a rubidium gas.
        PARAMS:     frac-> percent of Rb85/Rb87
                    temp-> temperature of the cell [K]
                    long-> length of the cell [m]
        RETURNS:    1darray
        """

        sum_alpha = self.alpha(10/81, 1-(frac/100), temp, omega, self.w_j, 12) + \
            self.alpha(35/81, 1-(frac/100), temp, self.w_jj, 12) + \
            self.alpha(1, 1-(frac/100), temp, omega, self.w_jjj, 12) + \
            self.alpha(1/3, 1-(frac/100), temp, omega, self.w_jb, 12) + \
            self.alpha(35/81, 1-(frac/100), temp, omega, self.w_jjb, 12) + \
            self.alpha(28/81, 1-(frac/100), temp, omega, self.w_jjjb, 12) + \
            self.alpha(1/18, frac/100, temp, self.w_i, 8) + \
            self.alpha(5/18, frac/100, temp, omega, self.w_ii, 8) + \
            self.alpha(7/9, frac/100, temp, omega, self.w_iii, 8) + \
            self.alpha(1/9, frac/100, temp, omega, self.w_ib, 8) + \
            self.alpha(5/18, frac/100, temp, omega, self.w_iib, 8) + \
            self.alpha(5/18, frac/100, temp, omega, self.w_iiib, 8)
        transmi = np.exp(-sum_alpha * long)
        return transmi
