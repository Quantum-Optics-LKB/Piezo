# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:01:54 2021

@author: Tangui ALADJIDI
"""

import numpy as np
import matplotlib.pyplot as plt
from ScopeInterface import USBScope, USBSpectrumAnalyzer
from piezo import PiezoTIM101

piezo = PiezoTIM101('65863991')
piezo.zero()
piezo.move(channel=1, pos=100)
piezo.disconnect()