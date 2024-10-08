# -*- coding: utf-8 -*-

import time
import traceback

import clr
import numpy as np

from .GenericDevice import GenericDevice

# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.Controls")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")

# Generic device manager
import Thorlabs.MotionControl.Controls
from System import Decimal
from Thorlabs.MotionControl.Benchtop.Piezo import *
from Thorlabs.MotionControl.Benchtop.PiezoCLI import *
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI import *


class BPC(GenericDevice):
    def __init__(self, serial: str = None) -> GenericDevice:
        """Instantiate a BPC object.

        This class is for controlling Thorlabs Benchtop Piezo Controllers with
        up to 3 channels.

        Args:
            serial (str, optional): The piezo serial number. Defaults to None.
        """
        self.device_prefix = BenchtopPiezo.DevicePrefix
        super().__init__(serial=serial, device_prefix=self.device_prefix)
        self.attempt_connection()
        self.channel_x = self.device.GetChannel("1")
        self.channel_y = self.device.GetChannel("2")
        self.channel_z = self.device.GetChannel("3")
        self.pos_x = float(str(self.channel_x.GetPosition()))
        self.pos_y = float(str(self.channel_y.GetPosition()))
        self.pos_z = float(str(self.channel_z.GetPosition()))
        # the GetJogSteps method returns
        # Thorlabs.MotionControl.GenericPiezoCLI.Settings.ControlSettings
        # object.
        # Need to the retrieve the PositionStepSize attribute to get a number
        self.step_x = float(str(self.channel_x.GetJogSteps().PositionStepSize))
        self.step_y = float(str(self.channel_y.GetJogSteps().PositionStepSize))
        self.step_z = float(str(self.channel_z.GetJogSteps().PositionStepSize))
        print("Device initialization successful !")
        for info in self.device_info:
            print(
                f"Device name : {info.Name}"
                + f", Serial Number : {info.SerialNumber}"
            )
        print(f"Number of channels : {self.device.ChannelCount}")
        print(
            f"Device position : x = {self.pos_x} µm"
            + f"y = {self.pos_y} µm, z = {self.pos_z} µm"
        )

    def attempt_connection(self):
        """Attempt to connect to the device.

        Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        """
        try:
            Thorlabs.MotionControl.Benchtop.PiezoCLI.BenchtopPiezo.ConnectDevice
            self.device = BenchtopPiezo.CreateBenchtopPiezo(self.serial)
            self.device.Connect(self.serial)
            deviceInfo = [self.device.GetDeviceInfo()]
            self.device_info = deviceInfo
            self.device.EnableCommsListener
        except Exception:
            print("ERROR : Could not connect to the device")
            print(traceback.format_exc())

    def get_position(self) -> float:
        """
        Gets the actual position of the piezo
        :return: (pos_x, pos_y, pos_z) a tuple of floats
        """
        self.pos_x = float(str(self.channel_x.GetPosition()))
        self.pos_y = float(str(self.channel_y.GetPosition()))
        self.pos_z = float(str(self.channel_z.GetPosition()))
        return self.pos_x, self.pos_y, self.pos_z

    def update_position(self):
        """
        Updates the position saved in the piezo object
        :return: None
        """
        self.pos_x, self.pos_y, self.pos_z = self.get_position()
        # need to find a way to actualize the position kept in
        # the python class regularly

    def move_to(self, tgt_x: float, tgt_y: float, tgt_z: float):
        """
        Moves the piezo to an arbitrary position
        :param tgt_x: x target position
        :param tgt_y: y target position
        :param tgt_z: z target position
        :return: None
        """
        self.update_position()
        # check if each argument is null, so that you don't need to specify
        # each argument if you want to move only one axis

        if tgt_x is not None:
            self.channel_x.SetPosition(Decimal(tgt_x))
        if tgt_y is not None:
            self.channel_y.SetPosition(Decimal(tgt_y))
        if tgt_z is not None:
            self.channel_z.SetPosition(Decimal(tgt_z))
        self.update_position()

    def sweep(self, axis: str = "x", rg: float = 1.0):
        """
        Does a 1D sweep along one axis
        :param axis: 'x', 'y' or 'z' the axis of the sweep
        :param rg: range of the sweep
        :return: None
        """

        self.update_position()
        if axis == "x":
            Range = np.arange(0, rg, self.step_x)
            start = self.pos_x - (rg / 2)
            origin = self.pos_x
            for x in Range:
                self.move_to(start + x, None, None)
                # do something
                time.sleep(0.5)
            self.move_to(origin, None, None)
            self.update_position()
        if axis == "y":
            Range = np.arange(0, rg, self.step_y)
            start = self.pos_y - (rg / 2)
            origin = self.pos_y
            for y in Range:
                self.move_to(None, start + y, None)
                # do something
                time.sleep(0.5)
            self.move_to(origin, None, None)
            self.update_position()
        if axis == "z":
            Range = np.arange(0, rg, self.step_z)
            start = self.pos_z - (rg / 2)
            origin = self.pos_z
            for z in Range:
                self.move_to(None, None, start + z)
                # do something
                time.sleep(0.5)
            self.move_to(origin, None, None)
            self.update_position()

    def sweep_2D(self, axis, rg0, rg1):
        """
        2D sweep in a plane (xy, yz or xz)
        :param axis: 'xy', 'yz' or 'xz' the plane in which the sweep is done
        :param rg0: Range along first axis
        :param rg1: Range along second axis
        :return: None
        """

        self.update_position()
        if axis == "xy":
            Range0 = np.arange(0, rg0, self.step_x)
            Range1 = np.arange(0, rg1, self.step_y)
            start_x = self.pos_x - (rg0 / 2)
            start_y = self.pos_y - (rg1 / 2)
            origin0 = self.pos_x
            origin1 = self.pos_y
            for x in Range0:
                for y in Range1:
                    print(f"Move to : x = {start_x + x} / y = {start_y + y}")
                    self.move_to(start_x + x, start_y + y, None)
                    # do something
                    time.sleep(0.5)
                    print(self.channel_x.GetPosition())
            self.move_to(origin0, origin1, None)
            self.update_position()
        if axis == "yz":
            Range0 = np.arange(0, rg0, self.step_y)
            Range1 = np.arange(0, rg1, self.step_z)
            start_y = self.pos_y - (rg0 / 2)
            start_z = self.pos_z - (rg1 / 2)
            origin0 = self.pos_y
            origin1 = self.pos_z
            for y in Range0:
                for z in Range1:
                    self.move_to(None, start_y + y, start_z + z)
                    # do something
                    time.sleep(0.5)
            self.move_to(None, origin0, origin1)
            self.update_position()
        if axis == "xz":
            Range0 = np.arange(0, rg0, self.step_x)
            Range1 = np.arange(0, rg1, self.step_z)
            start_x = self.pos_x - (rg0 / 2)
            start_z = self.pos_z - (rg1 / 2)
            origin0 = self.pos_x
            origin1 = self.pos_z
            for x in Range0:
                for z in Range1:
                    self.move_to(start_x + x, None, start_z + z)
                    # do something
                    time.sleep(0.5)
            self.move_to(origin0, None, origin1)
            self.update_position()
