# -*- coding: utf-8 -*-

"""
    Created on Fri Mar 29 11:00:03 2019

    @author: Tangui ALADJIDI

    Uses Kinesis to control Thorlabs devices

    The classes volontarily copy and wrap existing objects of Thorlabs
    .NET architecture to simplify it and bring useful objects to the
    relevant level in the inheritance tree.

"""

import clr
import sys
import os
import time
import numpy as np
import traceback

# VERY NAUGHTY : TO BE FIXED !!!!
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
clr.AddReference("Thorlabs.MotionControl.Controls")

from System import String, Decimal
# from System import Decimal
import System.Collections
from System.Collections import *
# Generic device manager
import Thorlabs.MotionControl.Controls
from Thorlabs.MotionControl.DeviceManagerCLI import *

from Thorlabs.MotionControl.Benchtop.PiezoCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *

from Thorlabs.MotionControl.TCube.InertialMotorCLI import *
from Thorlabs.MotionControl.TCube.DCServoCLI import *

from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *


class K10CR1:

    def __init__(self, serial: str = None):
        """Instantiates a K10CR1 object to control cage rotator

        :param str serial: Serial number
        :return: K10CR1 object

        """
        if serial is not None:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(CageRotator.DevicePrefix)
                if len(device_list) == 0:
                    print("Error : No K10CR1 motor found !")
                else:
                    if serial in device_list:
                        self.attempt_connection(serial)
                    else:
                        print("Error : Did not find the specified motor ")
                        for dev in device_list:
                            print(f"Device found, serial {dev}")
            except Exception:
                print("ERROR")
                print(traceback.format_exc())
        else:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(CageRotator.DevicePrefix)
                if len(device_list) == 0:
                    print("Error : No K10CR1 motor found !")
                elif len(device_list) == 1:
                    print("Only one device found, attempting to connect to " +
                          f"device {device_list[0]}")
                    self.attempt_connection(device_list[0])
                else:
                    for counter, dev in enumerate(device_list):
                        print(f"Device found, serial {dev} ({counter})")
                    choice = input("Choice (number between 0 and" +
                                   f" {len(device_list)-1})? ")
                    choice = int(choice)
                    self.attempt_connection(device_list[choice])
            except Exception:
                print("ERROR")
                print(traceback.format_exc())
        self.configuration = self.device.LoadMotorConfiguration(self.serial)
        self.settings = self.device.MotorDeviceSettings
        # do we home the device upon initialization ?
        # for task completion
        self.__taskID = 0
        self.__taskComplete = False

    def attempt_connection(self, serial: str):
        """Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        :param str serial: Serial number
        :return: None

        """
        try:
            self.device = CageRotator.CreateCageRotator(serial)
            self.device.Connect(serial)
            timeout = 0
            while not(self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            print("Success ! Connected to K10CR1 motor" +
                  f" {self.device_info.SerialNumber}" +
                  f" {self.device_info.Name}")
            self.serial = serial
        except Exception:
            print("ERROR : Could not connect to the device")
            print(traceback.format_exc())

    def disconnect(self):
        """Disconnects the device. Important for tidyness and to avoid
        references being kept to a dead object.

        :return: None

        """
        self.device.StopPolling()
        self.device.Disconnect(True)

    def __is_command_complete(self, taskID: int):
        """Private method to handle completion of tasks

        :param int taskID: Task whose status is being querried
        :return: None
        :rtype: Nonetype

        """
        if self.__taskID > 0 and self.__taskID == taskID:
            self.__taskComplete = True

    def home(self, timeout: float = 60e3) -> bool:
        """Homes the device to its center position. Might take some time.

        :param float timeout: Timeout of the movement.
        :return bool isHomed: If the device is homed

        """
        try:
            self.device.Home(int(timeout))
            print("Device homed !")
        except Exception:
            print("ERROR : Could not home the device")
            print(traceback.format_exc())
        # self.__taskComplete = false
        # Action = getattr(System, "Action`1")
        # checkcomplete = Action[UInt64](self.__is_command_complete)
        # self.__taskID = self.device.Home(checkcomplete)
        # t0 = time.time()
        # waittime = 0
        # while not(self.__taskComplete) and waittime < timeout:
        #     time.sleep(500e-3)
        #     status = self.device.Status
        #     sys.stdout.write(f"\rHoming device ... Position : {status.Position}")
        #     waittime = time.time()-t0

    def move_to(self, pos: float, timeout: float = 60e3):
        """Simple move

        :param float pos: Position
        :param float timeout: Timeout in ms to do the move
        :return: reached position
        :rtype: float

        """
        self.device.MoveTo(Decimal(pos), int(timeout))
        # self.device.SetMoveAbsolutePosition(Decimal(pos))
        # self.device.MoveAbsolute(int(timeout))
        print(f"Device position is: { self.device.Position } °.")
        return self.device.Position

    def get_position(self):
        """Returns the actual position

        :return: Position
        :rtype: float

        """
        print(f"Device position is: { self.device.Position } °.")
        return self.device.Status.Position



class TDC001:

    def __init__(self, serial: str = None):
        """Instantiates a TDC001 object to control piezo DC motor actuators

        :param str serial: Serial number
        :return: TDC001 object

        """
        if serial is not None:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(TCubeDCServo.DevicePrefix)
                if len(device_list) == 0:
                    print("Error : No TCube motor found !")
                else:
                    if serial in device_list:
                        self.attempt_connection(serial)
                    else:
                        print("Error : Did not find the specified motor ")
                        for dev in device_list:
                            print(f"Device found, serial {dev}")
            except Exception:
                print("ERROR")
                print(traceback.format_exc())
        else:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(TCubeDCServo.DevicePrefix)
                if len(device_list) == 0:
                    print("Error : No TCube motor found !")
                elif len(device_list) == 1:
                    print("Only one device found, attempting to connect to " +
                          f"device {device_list[0]}")
                    self.attempt_connection(device_list[0])
                else:
                    for counter, dev in enumerate(device_list):
                        print(f"Device found, serial {dev} ({counter})")
                    choice = input("Choice (number between 0 and" +
                                   f" {len(device_list)-1})? ")
                    choice = int(choice)
                    self.attempt_connection(device_list[choice])
            except Exception:
                print("ERROR")
                print(traceback.format_exc())
        self.configuration = self.device.LoadMotorConfiguration(self.serial)
        self.settings = self.device.MotorDeviceSettings
        # do we home the device upon initialization ?
        # for task completion
        self.__taskID = 0
        self.__taskComplete = False

    def attempt_connection(self, serial: str):
        """Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        :param str serial: Serial number
        :return: None

        """
        try:
            self.device = TCubeDCServo.CreateTCubeDCServo(serial)
            self.device.Connect(serial)
            timeout = 0
            while not(self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            print("Success ! Connected to TCube motor" +
                  f" {self.device_info.SerialNumber}" +
                  f" {self.device_info.Name}")
            self.serial = serial
        except Exception:
            print("ERROR : Could not connect to the device")
            print(traceback.format_exc())

    def disconnect(self):
        """Disconnects the device. Important for tidyness and to avoid
        references being kept to a dead object.

        :return: None

        """
        self.device.StopPolling()
        self.device.Disconnect(True)

    def __is_command_complete(self, taskID: int):
        """Private method to handle completion of tasks

        :param int taskID: Task whose status is being querried
        :return: None
        :rtype: Nonetype

        """
        if self.__taskID > 0 and self.__taskID == taskID:
            self.__taskComplete = True

    def home(self, timeout: float = 60e3) -> bool:
        """Homes the device to its center position. Might take some time.

        :param float timeout: Timeout of the movement.
        :return bool isHomed: If the device is homed

        """
        try:
            self.device.Home(int(timeout))
            print("Device homed !")
        except Exception:
            print("ERROR : Could not home the device")
            print(traceback.format_exc())
        # self.__taskComplete = false
        # Action = getattr(System, "Action`1")
        # checkcomplete = Action[UInt64](self.__is_command_complete)
        # self.__taskID = self.device.Home(checkcomplete)
        # t0 = time.time()
        # waittime = 0
        # while not(self.__taskComplete) and waittime < timeout:
        #     time.sleep(500e-3)
        #     status = self.device.Status
        #     sys.stdout.write(f"\rHoming device ... Position : {status.Position}")
        #     waittime = time.time()-t0

    def move_to(self, pos: float, timeout: float = 60e3):
        """Simple move

        :param float pos: Position
        :param float timeout: Timeout in ms to do the move
        :return: None
        :rtype: Nonetype

        """
        self.device.MoveTo(Decimal(pos), int(timeout))
        print(f"Device position is: { self.device.Position } °.")

    def get_position(self):
        """Returns the actual position

        :return: Position
        :rtype: float

        """
        print(f"Device position is: { self.device.Position } °.")
        return self.device.Status.Position



class BPC:
    def __init__(self, serial: float = None):
        """
        Instantiantes the piezo object
        :param serial: Thorlabs serial number
        """
        if serial is not None:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(BenchtopPiezo.DevicePrefix)
                if len(device_list) == 0:
                    print("Error : Could not find any device.")
                elif serial not in device_list:
                    print("Error : Could not find the specified device.")
                    for dev in device_list:
                        print(f"Device found, serial {dev}")
                else:
                    self.attempt_connection(serial)
            except Exception:
                print('ERROR')
                print(traceback.format_exc())
        else:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(BenchtopPiezo.DevicePrefix)
                if len(device_list) == 0:
                    print("Error : No Benchtop Piezo found !")
                elif len(device_list) == 1:
                    print("Only one device found, attempting to connect to " +
                          f"device {device_list[0]}")
                    self.attempt_connection(device_list[0])
                else:
                    for counter, dev in enumerate(device_list):
                        print(f"Device found, serial {dev} ({counter})")
                    choice = input("Choice (number between 0 and" +
                                   f" {len(device_list)-1})? ")
                    choice = int(choice)
                    self.attempt_connection(device_list[choice])
            except Exception:
                print("ERROR")
                print(traceback.format_exc())

        self.channel_x = self.device.GetChannel('1')
        self.channel_y = self.device.GetChannel('2')
        self.channel_z = self.device.GetChannel('3')
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
        print('Device initialization successful !')
        for info in self.device_info:
            print(f'Device name : {info.Name}' +
                  f', Serial Number : {info.SerialNumber}')
        print(f'Number of channels : {self.device.ChannelCount}')
        print(f'Device position : x = {self.pos_x} µm' +
              f'y = {self.pos_y} µm, z = {self.pos_z} µm')

    def attempt_connection(self, serial):
        """Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        :param str serial: Serial number
        :return: None

        """
        try:
            Thorlabs.MotionControl.Benchtop.PiezoCLI.BenchtopPiezo.ConnectDevice
            self.device = BenchtopPiezo.CreateBenchtopPiezo(serial)
            self.device.Connect(serial)
            deviceInfo = [self.device.GetDeviceInfo()]
            self.device_info = deviceInfo
            self.device.EnableCommsListener
            self.serial = serial
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

    def sweep(self, axis: str = 'x', rg: float = 1.0):
        """
        Does a 1D sweep along one axis
        :param axis: 'x', 'y' or 'z' the axis of the sweep
        :param rg: range of the sweep
        :return: None
        """

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
        """
        2D sweep in a plane (xy, yz or xz)
        :param axis: 'xy', 'yz' or 'xz' the plane in which the sweep is done
        :param rg0: Range along first axis
        :param rg1: Range along second axis
        :return: None
        """

        self.update_position()
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
                    self.move_to(None, start_y+y, start_z+z)
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


class TIM101:

    def __init__(self, serial: str = None):
        """Instantiates a TIM101 object to control piezo mirror screws

        :param str serial: Piezo serial number
        :return: PiezoScrew object
        :rtype: PiezoScrew

        """
        if serial is not None:
            DeviceManagerCLI.BuildDeviceList()
            device_list = DeviceManagerCLI.GetDeviceList(
                TCubeInertialMotor.DevicePrefix)
            if len(device_list) == 0 or serial not in device_list:
                print("Error : ")
            try:
                self.serial = serial  # SN of the Thorlabs Nano stage
                if len(device_list) == 0:
                    print("Error : No TCube motor found !")
                else:
                    if serial in device_list:
                        self.attempt_connection(serial)
                    else:
                        print("Error : Did not find the specified motor ")
                        for dev in device_list:
                            print(f"Device found, serial {dev}")
            except Exception:
                print("ERROR")
                print(traceback.format_exc())
        else:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(TCubeInertialMotor.DevicePrefix)
                if len(device_list) == 0:
                    print("Error : No TCube motor found !")
                elif len(device_list) == 1:
                    print("Only one device found, attempting to connect to " +
                          f"device {device_list[0]}")
                    self.attempt_connection(device_list[0])
                else:
                    for counter, dev in enumerate(device_list):
                        print(f"Device found, serial {dev} ({counter})")
                    choice = input("Choice (number between 0 and" +
                                   f" {len(device_list)-1})? ")
                    choice = int(choice)
                    self.attempt_connection(device_list[choice])
            except Exception:
                print("ERROR")
                print(traceback.format_exc())
        self.configuration = self.device.GetInertialMotorConfiguration(self.serial)
        self.settings = ThorlabsInertialMotorSettings.GetSettings(self.configuration)
        self.channel1 = InertialMotorStatus.MotorChannels.Channel1
        self.channel2 = InertialMotorStatus.MotorChannels.Channel2
        self.channel3 = InertialMotorStatus.MotorChannels.Channel3
        self.channel4 = InertialMotorStatus.MotorChannels.Channel4
        # set default settings StepRate and StepAcceleration
        self.settings.Drive.Channel(self.channel1).StepRate = 500
        self.settings.Drive.Channel(self.channel1).StepAcceleration = 100000
        self.settings.Drive.Channel(self.channel2).StepRate = 500
        self.settings.Drive.Channel(self.channel2).StepAcceleration = 100000
        self.settings.Drive.Channel(self.channel3).StepRate = 500
        self.settings.Drive.Channel(self.channel3).StepAcceleration = 100000
        self.settings.Drive.Channel(self.channel4).StepRate = 500
        self.settings.Drive.Channel(self.channel4).StepAcceleration = 100000
        self.device.SetSettings(self.settings, True, True)
        self.zero()

    def attempt_connection(self, serial):
        """Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        :param str serial: Serial number
        :return: None

        """
        try:
            self.device = TCubeInertialMotor.CreateTCubeInertialMotor(serial)
            self.device.Connect(serial)
            timeout = 0
            while not(self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            print("Success ! Connected to TCube motor" +
                  f" {self.device_info.SerialNumber}" +
                  f" {self.device_info.Name}")
            self.serial = serial
        except Exception:
            print("ERROR : Could not connect to the device")
            print(traceback.format_exc())

    def get_steprate(self, channel: int = 1) -> float:
        """
        Wrapper function to get the step rate of a channel in Hz
        :param channel: Channel number
        :return: steprate
        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            self.settings = ThorlabsInertialMotorSettings.GetSettings(self.configuration)
            if channel == 1:
                return self.settings.Drive.Channel(self.channel1).StepRate
            elif channel == 2:
                return self.settings.Drive.Channel(self.channel2).StepRate
            elif channel == 3:
                return self.settings.Drive.Channel(self.channel3).StepRate
            elif channel == 4:
                return self.settings.Drive.Channel(self.channel4).StepRate

    def get_stepaccel(self, channel: int = 1) -> float:
        """
        Wrapper function to get the step acceleration in Hz^2
        :param channel: channel number
        :return: None
        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            self.settings = ThorlabsInertialMotorSettings.GetSettings(self.configuration)
            if channel == 1:
                return self.settings.Drive.Channel(self.channel1).StepAcceleration
            elif channel == 2:
                return self.settings.Drive.Channel(self.channel2).StepAcceleration
            elif channel == 3:
                return self.settings.Drive.Channel(self.channe13).StepAcceleration
            elif channel == 4:
                return self.settings.Drive.Channel(self.channel4).StepAcceleration

    def set_steprate(self, channel: int = 1, steprate: int = 500):
        """
        Sets the step rate
        :param channel: Channel number
        :param steprate: Step rate in Hz
        :return: None
        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            self.settings = ThorlabsInertialMotorSettings.GetSettings(self.configuration)
            if channel == 1:
                self.settings.Drive.Channel(self.channel1).StepRate = steprate
            elif channel == 2:
                self.settings.Drive.Channel(self.channel2).StepRate = steprate
            elif channel == 3:
                self.settings.Drive.Channel(self.channel3).StepRate = steprate
            elif channel == 4:
                self.settings.Drive.Channel(self.channel4).StepRate = steprate
            self.device.SetSettings(self.settings, True, True)

    def set_stepaccel(self, channel: int = 1, stepaccel: int = 100000):
        """
        Sets the step acceleration
        :param channel: Channel number
        :param stepaccel: Step acceleration in Hz^2
        :return: None
        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            self.settings = ThorlabsInertialMotorSettings.GetSettings(self.configuration)
            if channel == 1:
                self.settings.Drive.Channel(self.channel1).StepAcceleration = stepaccel
            elif channel == 2:
                self.settings.Drive.Channel(self.channel2).StepAcceleration = stepaccel
            elif channel == 3:
                self.settings.Drive.Channel(self.channel3).StepAcceleration = stepaccel
            elif channel == 4:
                self.settings.Drive.Channel(self.channel4).StepAcceleration = stepaccel
            self.device.SetSettings(self.settings, True, True)

    def zero(self, channel: int = 1):
        """
        Zeros the piezo's position at its current position
        :param channel: Channel number
        :return: None
        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            if channel == 1:
                self.device.SetPositionAs(self.channel1, 0)
            elif channel == 2:
                self.device.SetPositionAs(self.channel2, 0)
            elif channel == 3:
                self.device.SetPositionAs(self.channel3, 0)
            elif channel == 4:
                self.device.SetPositionAs(self.channel4, 0)

    def get_position(self, channel: int = 1) -> int:
        """Retrieves the piezo's current position

        :param int channel: Channel number
        :return: Position of the piezo in steps
        :rtype: int

        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            if channel == 1:
                pos = self.device.GetPosition(self.channel1)
            elif channel == 2:
                pos = self.device.GetPosition(self.channel2)
            elif channel == 3:
                pos = self.device.GetPosition(self.channel3)
            elif channel == 4:
                pos = self.device.GetPosition(self.channel4)
            return pos

    def move_to(self, channel: int = 1, pos: int = 0) -> int:
        """
        Moves the piezo to a specified position
        :param channel: Channel number
        :param pos: Position (int)
        :return: Current Position
        :rtype: int
        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            try:
                if channel == 1:
                    self.device.MoveTo(self.channel1, pos, 120000)
                elif channel == 2:
                    self.device.MoveTo(self.channel2, pos, 120000)
                elif channel == 3:
                    self.device.MoveTo(self.channel3, pos, 120000)
                elif channel == 4:
                    self.device.MoveTo(self.channel4, pos, 120000)
            except Exception:
                print("ERROR : Failed to move")
                print(traceback.format_exc())
            curr_pos = self.get_position(channel)
            sys.stdout.write(f"\r Moved to : {curr_pos}")
            return curr_pos

    def disconnect(self):
        """
        Wrapper function to disconnect the object. Important for tidyness and to avoid conflicts with
        Kinesis
        :return: None
        """
        self.device.StopPolling()
        self.device.Disconnect(True)
