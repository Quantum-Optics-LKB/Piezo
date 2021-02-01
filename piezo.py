# -*- coding: utf-8 -*-

"""
    Created on Fri Mar 29 11:00:03 2019

    @author: tangui

    Uses Kinesis to control Thorlabs piezo stage in XYZ configuration
    Or uses the DAISY DLL's to control Attocube stage
    https://github.com/Laukei/attocube-ANC350-Python-library

"""

import clr
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import inspect
import traceback

from System import String
from System import Decimal
import System.Collections
from System.Collections import *
# for Attocube
# from pyanc350 import PyANC350v4


# constants
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
# needs local Kinesis folder, but maybe it is possible to automatically
# find it
# add .net reference and import so python can see .net
clr.AddReference("Thorlabs.MotionControl.Controls")
import Thorlabs.MotionControl.Controls


# Add references so Python can see .Net
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")


from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.Benchtop.PiezoCLI import *


# add this, see examples from kinesis
from Thorlabs.MotionControl.GenericPiezoCLI import *
# print(dir(Thorlabs.MotionControl.DeviceManagerCLI))
# print(dir(Thorlabs.MotionControl.Benchtop.PiezoCLI))
# print(dir(Thorlabs.MotionControl.Benchtop.PiezoCLI.PiezoChannel))
# print(dir(Thorlabs.MotionControl.Benchtop.PiezoCLI.BenchtopPiezo))
# print(dir(Thorlabs.MotionControl.Benchtop.PiezoCLI.BenchtopPiezoConfiguration))

# thorlabs nanostage serial numbers of each axis '71897085-1' to -3


# print(Thorlabs.MotionControl.GenericPiezoCLI.__doc__)


# Main Piezo class that contains device infos, and basic movement routines for Thorlabs stage
class Piezo3Axis():
    # initializes contact with the device
    def __init__(self, serial):
        try:
            self.serial = serial  # SN of the Thorlabs Nano stage
            Thorlabs.MotionControl.Benchtop.PiezoCLI.BenchtopPiezo.ConnectDevice
            self.device = BenchtopPiezo.CreateBenchtopPiezo(serial)
            device_list_result = DeviceManagerCLI.BuildDeviceList()
            self.device.Connect(serial)
            deviceInfo = [self.device.GetDeviceInfo()]
            self.device_info = deviceInfo
            self.device.EnableCommsListener
            self.channel_x = self.device.GetChannel('1')
            self.channel_y = self.device.GetChannel('2')
            self.channel_z = self.device.GetChannel('3')
            self.pos_x = float(str(self.channel_x.GetPosition()))
            self.pos_y = float(str(self.channel_y.GetPosition()))
            self.pos_z = float(str(self.channel_z.GetPosition()))
            # the GetJogSteps method returns Thorlabs.MotionControl.GenericPiezoCLI.Settings.ControlSettings object.
            # Need to the retrieve the PositionStepSize attribute to get a number
            self.step_x = float(str(self.channel_x.GetJogSteps().PositionStepSize))
            self.step_y = float(str(self.channel_y.GetJogSteps().PositionStepSize))
            self.step_z = float(str(self.channel_z.GetJogSteps().PositionStepSize))
            print('Device initialization successful !')
            for info in deviceInfo:
                print('Device name : ',info.Name, '/  Serial Number : ', info.SerialNumber)
            print('Number of channels : ',self.device.ChannelCount)
            print('Device position : x = ',self.pos_x,' µm, y = ', self.pos_y,' µm, z = ',self.pos_z,' µm')
        except Exception:
            print('Could not connect with the device ... Have you checked that it is not already used by another program ?')
            print(traceback.format_exc())

    def get_position(self):
        self.pos_x = float(str(self.channel_x.GetPosition()))
        self.pos_y = float(str(self.channel_y.GetPosition()))
        self.pos_z = float(str(self.channel_z.GetPosition()))
        return self.pos_x, self.pos_y, self.pos_z

    def update_position(self):
        self.pos_x, self.pos_y, self.pos_z = self.get_position() #need to find a way to actualize the position kept in
                                                                 #the python class regularly

    # simple move method : move to an arbitrary point
    def move_to(self, tgt_x, tgt_y, tgt_z):
        self.update_position()
        # check if each argument is null, so that you don't need to specify each
        # argument if you want to move only one axis

        if tgt_x is not None:
            self.channel_x.SetPosition(Decimal(tgt_x))
        if tgt_y is not None:
            self.channel_y.SetPosition(Decimal(tgt_y))
        if tgt_z is not None:
            self.channel_z.SetPosition(Decimal(tgt_z))
        self.update_position()

    def sweep(self, axis, rg):
        # method to sweep over a range rg on the axis 'x', 'y', or 'z' from the starting position
        # actualize position if ever it has not been done before
        self.update_position()
        if axis == 'x':
            Range = np.arange(0, rg, self.step_x)
            start = self.pos_x-(rg/2)
            origin = self.pos_x
            for x in Range:
                self.move_to(start+x, None, None)
                # do something
                time.sleep(0.5)
            self.move_to(origin, None, None)
            self.update_position()
        if axis == 'y':
            Range = np.arange(0, rg, self.step_y)
            start = self.pos_y-(rg/2)
            origin = self.pos_y
            for y in Range:
                self.move_to(None, start+y, None)
                # do something
                time.sleep(0.5)
            self.move_to(origin, None, None)
            self.update_position()
        if axis == 'z':
            Range = np.arange(0, rg, self.step_z)
            start = self.pos_z-(rg/2)
            origin = self.pos_z
            for z in Range:
                self.move_to(None, None, start+z)
                # do something
                time.sleep(0.5)
            self.move_to(origin, None, None)
            self.update_position()

    def sweep_2D(self, axis, rg0, rg1):
        self.update_position()
        # method to sweep a surface (xy,yz, or xz) over a range rg0 on the
        # first axis and rg1 on the other
        if axis == 'xy':
            Range0 = np.arange(0, rg0, self.step_x)
            Range1 = np.arange(0, rg1, self.step_y)
            start_x = self.pos_x-(rg0/2)
            start_y = self.pos_y-(rg1/2)
            origin0 = self.pos_x
            origin1 = self.pos_y
            for x in Range0:
                for y in Range1:
                    print(f"Move to : x = {start_x+x} / y = {start_y+y}")
                    self.move_to(start_x+x, start_y+y, None)
                    # do something
                    time.sleep(0.5)
                    print(self.channel_x.GetPosition())
           self.move_to(origin0, origin1, None)
           self.update_position()
       if axis == 'yz':
           Range0 = np.arange(0, rg0, self.step_y)
           Range1 = np.arange(0, rg1, self.step_z)
           start_y = self.pos_y - (rg0/2)
           start_z = self.pos_z - (rg1/2)
           origin0 = self.pos_y
           origin1 = self.pos_z
           for y in Range0:
               for z in Range1:
                   self.move_to(None , start_y+y, start_z+z)
                   # do something
                   time.sleep(0.5)
           self.move_to(None, origin0, origin1)
           self.update_position()
       if axis == 'xz':
           Range0 = np.arange(0, rg0, self.step_x)
           Range1 = np.arange(0, rg1, self.step_z)
           start_x = self.pos_x - (rg0/2)
           start_z = self.pos_z - (rg1/2)
           origin0 = self.pos_x
           origin1 = self.pos_z
           for x in Range0:
               for z in Range1:
                   self.move_to(start_x+x, None, start_z+z)
                   # do something
                   time.sleep(0.5)
           self.move_to(origin0, None, origin1)
           self.update_position()
class PiezoScrew:
    def __init__(self, serial: str = None):
        """Instantiates a PiezoScrew object to control piezo mirror screws

        :param str serial: Piezo serial number
        :return: PiezoScrew object
        :rtype: PiezoScrew

        """
        try:
            self.serial = serial  # SN of the Thorlabs Nano stage
            Thorlabs.MotionControl.Benchtop.PiezoCLI.BenchtopPiezo.ConnectDevice
            self.device = BenchtopPiezo.CreateBenchtopPiezo(serial)
            device_list_result = DeviceManagerCLI.BuildDeviceList()
            self.device.Connect(serial)
            deviceInfo = [self.device.GetDeviceInfo()]
            self.device_info = deviceInfo
            self.device.EnableCommsListener
            self.channel = self.device.GetChannel('1')
            self.pos = float(str(self.channel.GetPosition()))
            # the GetJogSteps method returns Thorlabs.MotionControl.GenericPiezoCLI.Settings.ControlSettings object.
            # Need to the retrieve the PositionStepSize attribute to get a number
            self.step = float(str(self.channel.GetJogSteps().PositionStepSize))
            print('Device initialization successful !')
            for info in deviceInfo:
                print('Device name : ',info.Name, '/  Serial Number : ', info.SerialNumber)
            print('Number of channels : ',self.device.ChannelCount)
            print(f"Device position : x = {self.pos_x} µm")
        except Exception:
            print('Could not connect with the device ... Have you checked that it is not already used by another program ?')
            print(traceback.format_exc())
