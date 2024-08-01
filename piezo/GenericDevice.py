# -*- coding: utf-8 -*-
import sys
from typing import Any

import clr

# import ctypes

# VERY NAUGHTY : TO BE FIXED !!!!
# try:
#     path = ctypes.util.find_library("Kinesis")
# except FileNotFoundError:
#     print("Error : Kinesis library not found")
#     sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")

from System.Collections import *

# Generic device manager
from Thorlabs.MotionControl.DeviceManagerCLI import *


class GenericDevice:
    """
    A generic class to control Thorlabs devices using the Kinesis .NET API.
    """

    def __init__(self, serial: str = None, device_prefix: Any = None):
        """Instantiate a generic object connecting to the device.

        Args:
            serial (str, optional): Device serial number. Defaults to None.
            device_prefix (Any, optional): Kinesis device prefix.
                Defaults to None.
                For example: CageRotator.DevicePrefix
        """
        self.serial = serial
        self.device_prefix = device_prefix
        # build the device list
        DeviceManagerCLI.BuildDeviceList()
        if self.device_prefix is None:
            device_list = DeviceManagerCLI.GetDeviceList()
        else:
            device_list = DeviceManagerCLI.GetDeviceList(self.device_prefix)
        if len(device_list) == 0:
            raise ConnectionError("No device found")
        else:
            for counter, dev in enumerate(device_list):
                print(f"Device found, serial {dev} ({counter})")
        if self.serial is None:
            choice = input(
                "Choice (number between 0 and" + f" {len(device_list) - 1})? "
            )
            choice = int(choice)
            self.serial = device_list[choice]
        if self.serial not in device_list:
            raise ConnectionError("Device not found")

    def disconnect(self):
        """
        Wrapper function to disconnect the object.
        Important for tidyness and to avoid conflicts with
        Kinesis
        :return: None
        """
        self.device.StopPolling()
        self.device.Disconnect(True)
