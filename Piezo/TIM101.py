# -*- coding: utf-8 -*-
import sys
import time
import traceback

import clr

from .GenericDevice import GenericDevice

# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")

# TODO Maybe useless imports, but this needs to be checked.
from System.Collections import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *
from Thorlabs.MotionControl.TCube.DCServoCLI import *
from Thorlabs.MotionControl.TCube.InertialMotorCLI import *


class TIM101(GenericDevice):
    def __init__(self, serial: str = None) -> object:
        """Instantiates a TIM101 object to control piezo mirror screws

        :param str serial: Piezo serial number
        :return: PiezoScrew object
        :rtype: PiezoScrew

        """
        self.device_prefix = TCubeInertialMotor.DevicePrefix
        super().__init__(serial=serial, device_prefix=self.device_prefix)
        self.configuration = self.device.GetInertialMotorConfiguration(
            self.serial
        )
        self.settings = ThorlabsInertialMotorSettings.GetSettings(
            self.configuration
        )
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

    def attempt_connection(self) -> None:
        """Attempt connection to the device.

        Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        """
        try:
            self.device = TCubeInertialMotor.CreateTCubeInertialMotor(
                self.serial
            )
            self.device.Connect(self.serial)
            timeout = 0
            while not (self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            print(
                "Success ! Connected to TCube motor"
                + f" {self.device_info.SerialNumber}"
                + f" {self.device_info.Name}"
            )
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
            self.settings = ThorlabsInertialMotorSettings.GetSettings(
                self.configuration
            )
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
            self.settings = ThorlabsInertialMotorSettings.GetSettings(
                self.configuration
            )
            if channel == 1:
                return self.settings.Drive.Channel(
                    self.channel1
                ).StepAcceleration
            elif channel == 2:
                return self.settings.Drive.Channel(
                    self.channel2
                ).StepAcceleration
            elif channel == 3:
                return self.settings.Drive.Channel(
                    self.channe13
                ).StepAcceleration
            elif channel == 4:
                return self.settings.Drive.Channel(
                    self.channel4
                ).StepAcceleration

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
            self.settings = ThorlabsInertialMotorSettings.GetSettings(
                self.configuration
            )
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
            self.settings = ThorlabsInertialMotorSettings.GetSettings(
                self.configuration
            )
            if channel == 1:
                self.settings.Drive.Channel(
                    self.channel1
                ).StepAcceleration = stepaccel
            elif channel == 2:
                self.settings.Drive.Channel(
                    self.channel2
                ).StepAcceleration = stepaccel
            elif channel == 3:
                self.settings.Drive.Channel(
                    self.channel3
                ).StepAcceleration = stepaccel
            elif channel == 4:
                self.settings.Drive.Channel(
                    self.channel4
                ).StepAcceleration = stepaccel
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
