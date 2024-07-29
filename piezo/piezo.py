# -*- coding: utf-8 -*-

import clr
import sys
import time
import numpy as np
import traceback
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
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
clr.AddReference("Thorlabs.MotionControl.Controls")

from System import String, Decimal
import System.Collections
from System.Collections import *
# Generic device manager
import Thorlabs.MotionControl.Controls
from Thorlabs.MotionControl.DeviceManagerCLI import *

from Thorlabs.MotionControl.Benchtop.Piezo import *
from Thorlabs.MotionControl.Benchtop.PiezoCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *

from Thorlabs.MotionControl.TCube.InertialMotorCLI import *
from Thorlabs.MotionControl.TCube.DCServoCLI import *

from Thorlabs.MotionControl.KCube.InertialMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *

from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *

from Thorlabs.MotionControl.Benchtop.PiezoCLI.PDXC2 import *
from Thorlabs.MotionControl.Benchtop.Piezo.PDXC2 import *
from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes


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
        sys.stdout.write(f"\rDevice position is: {self.device.Position} °.")
        # WARNING : This is ugly ! System decimal separator needs to be set to
        # "." for it to work !!! 
        return float(str(self.device.Position))

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
        self.short_name = None # Short name of actuator, assigned later if connection successful

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
        print(f"{self.short_name} | Configured for {self.configuration.DeviceSettingsName} stage")
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
            self.short_name = self.device_info.Name
            print("Success ! Connected to TCube motor:" +
                  f" {self.device_info.Description}" + 
                  f", S/N: {self.device_info.SerialNumber}"
                  )
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
        print(f"{self.short_name} | Homing device...")
        try:
            self.device.Home(int(timeout))
            print(f"{self.short_name} | Device homed !")
        except Exception:
            print(f"{self.short_name} | ERROR : Could not home the device")
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
        """Simple move, returns final position in device coordinates

        :param float pos: Position
        :param float timeout: Timeout in ms to do the move
        :return: None
        :rtype: float

        """
        print(f"{self.short_name} | Moving to {pos}°")
        self.device.MoveTo(Decimal(pos), int(timeout))
        print(f"{self.short_name} | Device position: { self.device.Position }°")
        return float(str(self.device.Position))

    def get_position(self):
        """Returns the actual position

        :return: Position
        :rtype: float

        """
        print(f"{self.short_name} | Device position: { self.device.Position }°")
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


class KIM101:

    def __init__(self, serial: str = None):
        """Instantiates a TIM101 object to control piezo mirror screws

        :param str serial: Piezo serial number
        :return: PiezoScrew object
        :rtype: PiezoScrew

        """
        if serial is not None:
            DeviceManagerCLI.BuildDeviceList()
            device_list = DeviceManagerCLI.GetDeviceList(
                KCubeInertialMotor.DevicePrefix_KIM101)
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
                device_list = DeviceManagerCLI.GetDeviceList(KCubeInertialMotor.DevicePrefix_KIM101)
                if len(device_list) == 0:
                    print("Error : No KCube motor found !")
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
        self.settings.Drive.Channel(self.channel1).StepRate = 2000
        self.settings.Drive.Channel(self.channel1).StepAcceleration = 100000
        self.settings.Drive.Channel(self.channel2).StepRate = 2000
        self.settings.Drive.Channel(self.channel2).StepAcceleration = 100000
        self.settings.Drive.Channel(self.channel3).StepRate = 2000
        self.settings.Drive.Channel(self.channel3).StepAcceleration = 100000
        self.settings.Drive.Channel(self.channel4).StepRate = 2000
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
            self.device = KCubeInertialMotor.CreateKCubeInertialMotor(serial)
            self.device.Connect(serial)
            timeout = 0
            while not(self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            print("Success ! Connected to KCube motor" +
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

    def move_to(self, channel: int = 1, pos: int = 0, timeout: int = 2000) -> int:
        """
        Moves the piezo to a specified position
        :param channel: Channel number
        :param pos: Position (int)
        :param timeout: Timeout in ms defaults to 2000
        :return: Current Position
        :rtype: int
        """
        if channel not in [1, 2, 3, 4]:
            print("Error : Channel number must be between 1 and 4")
        else:
            try:
                if channel == 1:
                    self.device.MoveTo(self.channel1, int(pos), int(timeout))
                elif channel == 2:
                    self.device.MoveTo(self.channel2, int(pos), int(timeout))
                elif channel == 3:
                    self.device.MoveTo(self.channel3, int(pos), int(timeout))
                elif channel == 4:
                    self.device.MoveTo(self.channel4, int(pos), int(timeout))
            except Exception:
                print("ERROR : Failed to move")
                print(traceback.format_exc())
            curr_pos = self.get_position(channel)
            sys.stdout.write(f"\r Moved to : {curr_pos:06d}")
            return curr_pos

    def disconnect(self):
        """
        Wrapper function to disconnect the object. Important for tidyness and to avoid conflicts with
        Kinesis
        :return: None
        """
        self.device.StopPolling()
        self.device.Disconnect(True)


class KDC101:
    def __init__(self, serial: str = None, stage_name: str = 'Z825'):
        """Instantiates a TIM101 object to control piezo mirror screws

        :param str serial: Piezo serial number
        :param str stage_name: Stage type that is connected to the controller
        this allows conversion of the move units to real units.
        :return: PiezoScrew object
        :rtype: PiezoScrew

        """
        if serial is not None:
            DeviceManagerCLI.BuildDeviceList()
            device_list = DeviceManagerCLI.GetDeviceList(
                KCubeDCServo.DevicePrefix)
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
                device_list = DeviceManagerCLI.GetDeviceList(KCubeDCServo.DevicePrefix)
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
        # for real world units conversion
        self.configuration.DeviceSettingsName = stage_name
        self.configuration.UpdateCurrentConfiguration()
        # velparams = self.device.GetVelocityParams()
        # velparams.MaxVelocity = Decimal(4.0)
        # self.device.SetVelocityParams(velparams)
        # set default settings 
        self.settings = self.device.MotorDeviceSettings
        self.device.SetSettings(self.settings, True, True)
        # velparams = self.device.GetVelocityParams()
        # print(velparams.MaxVelocity)


    def attempt_connection(self, serial):
        """Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        :param str serial: Serial number
        :return: None

        """
        try:
            self.device = KCubeDCServo.CreateKCubeDCServo(serial)
            self.device.Connect(serial)
            timeout = 0
            while not(self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            print("Success ! Connected to KCube motor" +
                  f" {self.device_info.SerialNumber}" +
                  f" {self.device_info.Name}")
            self.serial = serial
        except Exception:
            print("ERROR : Could not connect to the device")
            print(traceback.format_exc())

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
        sys.stdout.write(f"\rDevice position is: {self.device.Position} mm.")
        # WARNING : This is ugly ! System decimal separator needs to be set to
        # "." for it to work !!! 
        return float(str(self.device.Position))

    def disconnect(self):

        """
        Wrapper function to disconnect the object. Important for tidyness and to avoid conflicts with
        Kinesis
        :return: None
        """
        self.device.StopPolling()
        self.device.Disconnect(True)


class LTS:
   
    def __init__(self, serial: str = None):
        """Instantiates a K10CR1 object to control cage rotator

        :param str serial: Serial number
        :return: K10CR1 object

        """
        if serial is not None:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(LongTravelStage.DevicePrefix)
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
                device_list = DeviceManagerCLI.GetDeviceList(LongTravelStage.DevicePrefix)
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
        # Sets the velocities and acceleration to their maximum values
        velParams = self.device.GetVelocityParams()
        velParams.Acceleration = Decimal(10)
        velParams.MaxVelocity = Decimal(15)
        velParams.MinVelocity = Decimal(0)
        self.device.SetVelocityParams(velParams)

    def attempt_connection(self, serial: str):
        """Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        :param str serial: Serial number
        :return: None

        """
        try:
            self.device = LongTravelStage.CreateLongTravelStage(serial)
            self.device.Connect(serial)
            timeout = 0
            while not(self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            print("Success ! Connected to LTS " +
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

    def move_to(self, pos: float, timeout: float = 60e3):
        """Simple move

        :param float pos: Position
        :param float timeout: Timeout in ms to do the move
        :return: reached position
        :rtype: float

        """
        self.device.MoveTo(Decimal(pos), int(timeout))
        sys.stdout.write(f"\rDevice position is: {self.device.Position} mm.")
        # WARNING : This is ugly ! System decimal separator needs to be set to
        # "." for it to work !!! 
        return float(str(self.device.Position))

    def get_position(self):
        """Returns the actual position

        :return: Position
        :rtype: float

        """
        print(f"Device position is: { self.device.Position } °.")
        return self.device.Status.Position
    

class PDXC2:

    def __init__(self, serial: str = None, mode: int = 2):
        """Instantiates a PDXC2 object to control piezo stage.

        :param str serial: Serial number
        :param int mode: OpenLoop (1) or CloseLoop (2)
        :return: PDXC2 object

        """
        self.short_name = None # Short name of actuator, assigned later if connection successful

        if serial is not None:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(InertiaStageController.DevicePrefix_PDXC2)
                if len(device_list) == 0:
                    print("Error : Could not find any device.")
                elif serial not in device_list:
                    print("Error : Could not find the specified device.")
                    for dev in device_list:
                        print(f"Device found, serial {dev}")
                else:
                    print('Attempt connection ...')
                    self.attempt_connection(serial, mode)
                    
            except Exception as err:
                raise err
        else:
            try:
                DeviceManagerCLI.BuildDeviceList()
                device_list = DeviceManagerCLI.GetDeviceList(InertiaStageController.DevicePrefix_PDXC2)
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
                    self.attempt_connection(device_list[choice], mode)
            except Exception as err:
                raise err
            
        self.position = self.get_position()
        self.control_mode = self.get_position_control_mode()
        
    def attempt_connection(self, serial, mode):
        """Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        :param str serial: Serial number
        :return: None

        """
        try:
            self.device = InertiaStageController.CreateInertiaStageController(serial)
            self.device.Connect(serial)
            timeout = 0
            while not(self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
                if timeout == 10:
                    raise RuntimeError('Settings failed to initialize.')
                
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            time.sleep(0.1)
            deviceInfo = [self.device.GetDeviceInfo()]
            self.device_info = deviceInfo
            self.device.EnableCommsListener
            print("Success ! Connected to PDXC2 piezo:" +
                  f" {self.device_info[0].Description}" + 
                  f", S/N: {self.device_info[0].SerialNumber}"
                  )
            self.serial = serial

            self.configuration = self.device.GetPDXC2Configuration(serial, DeviceConfiguration.DeviceSettingsUseOptionType.UseDeviceSettings)
            self.settings = PDXC2Settings.GetSettings(self.configuration)
            self.device.SetSettings(self.settings, True, True)

            if mode == 1:
                self.set_open_loop()
            elif mode == 2:
                self.set_close_loop()
            else:
                self.set_close_loop()
                raise UserWarning('1 (OpenLoop) or 2 (CloseLoop) must be provided. CloseLoop mode set by default.')
        except Exception:
            print("ERROR : Could not connect to the device")
            print(traceback.format_exc())

    def get_position(self) -> int:
        """
        Gets the actual position of the piezo.
        :return: int
        """
        self.device.RequestCurrentPosition()
        self.position = self.device.GetCurrentPosition()
        return self.position
    
    def update_position(self):
        """
        Updates the position saved in the piezo object
        :return: None
        """
        self.position = self.get_position()

    def get_position_control_mode(self):
        """
        Gets the actual position control mode (Open or Close loop).
        :return: PiezoControlModeTypes 
        """
        self.control_mode = self.device.GetPositionControlMode()
        return self.control_mode
    
    def set_open_loop(self):
        """
        Changes the position control mode to Open Loop. 
        :return: None
        """
        self.device.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)
        self.control_mode = PiezoControlModeTypes.OpenLoop
        print("Device set to OpenLoop mode.")

    def set_close_loop(self):
        """
        Changes the position control mode to Closed Loop. 
        :return: None
        """
        self.device.SetPositionControlMode(PiezoControlModeTypes.CloseLoop)
        self.control_mode = PiezoControlModeTypes.CloseLoop
        print("Device set to CloseLoop mode.")

    def move_to(self, tgt: float):
        """
        Moves the piezo to an arbitrary position
        :param tgt: Float
        :return: None
        """
        self.update_position()
        if self.get_position_control_mode() == PiezoControlModeTypes.CloseLoop:
            print(f'Closed loop target position: {tgt}')
            try:
                self.device.SetClosedLoopTarget(tgt)
                self.device.MoveStart()
                print(f"Device moved to position: {tgt}.")
            except Exception:
                print("Fail to move to the target position.")
        else:
            print(f'Open loop target position: {tgt}')
            openloop_params = self.device.GetOpenLoopMoveParameters()
            openloop_params.set_StepSize(tgt)
            self.device.SetOpenLoopMoveParameters(openloop_params)
            try:
                self.device.MoveStart()
                new_position = self.get_position()
                timeout = 0
                while tgt != new_position and timeout < 100:
                    new_position = self.get_position()
                    timeout+=1
                print(f"Device moved to position: {tgt}.")
            except Exception:
                print("Fail to move to the target position.")

    def move_by(self, dist: float):
        """
        Moves the piezo by a distance. 
        :param dist: Float
        :return: None
        """
        self.update_position()
        if self.get_position_control_mode() == PiezoControlModeTypes.CloseLoop:
            try:
                print(f'Closed loop distance set: {dist}')
                self.device.SetClosedLoopTarget(int(self.get_position()) + dist)
                self.device.MoveStart()
                print(f"Device moved by a distance: {dist}.")
            except Exception as err:
                print(f"Fail to move by a distance: {dist}.")
                print(err)
        else:
            pass 
            ##TODO 
        
    def jog(self, direction: int = 1, stepsize: int = 10):
        """
        Jog piezo. 
        :param: direction: int, stepsize: int
        :return: None
        """
        if self.get_position_control_mode() == PiezoControlModeTypes.CloseLoop:
            jogparams = self.device.GetJogParameters()
            jogparams.set_ClosedLoopStepSize(stepsize)
            jogparams.set_Mode(JogParameters.JogModes.Step)
            self.device.SetJogParameters(jogparams)
        else:
            jogparams = self.device.GetJogParameters()
            jogparams.set_OpenLoopStepSize(stepsize)
            jogparams.set_Mode(JogParameters.JogModes.Step)
            self.device.SetJogParameters(jogparams)
        
        if direction == 0:
            self.device.Jog(PDXC2TravelDirection(0), None)
        elif direction == 1:
            self.device.Jog(PDXC2TravelDirection(1), None)
        elif direction == -1:
            self.device.Jog(PDXC2TravelDirection(2), None)
        else:
            raise ValueError('Direction parameter must be 1 for forward, -1 for reverse or 0 for "undefined".')        
        self.update_position()

    def home(self):
        """
        Moves the piezo to the home position. In open loop mode, it sets the 0 position to the current position.
        :return: None
        """
        self.device.Home(int(60e3))
        self.update_position()
        print('Device homed.')

    def disconnect(self):
        """
        Disconnects the device.
        :return: None
        """
        self.device.DisableDevice()
        self.device.StopPolling()
        self.device.Disconnect(True)
        print(f'Device {self.device_info[0].SerialNumber} deconnected.')
        
