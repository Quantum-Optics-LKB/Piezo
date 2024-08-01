# -*- coding: utf-8 -*-

import sys
import time
import traceback

import clr

from .GenericDevice import GenericDevice

# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
clr.AddReference("Thorlabs.MotionControl.Controls")


# Generic device manager
from System import Decimal
from System.Collections import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *
from Thorlabs.MotionControl.KCube.InertialMotorCLI import *


class KDC101(GenericDevice):
    def __init__(
        self, serial: str = None, stage_name: str = "Z825"
    ) -> GenericDevice:
        """Instantiate a KDC101 object.

        This controls a KCube DC Servo motor.
        In order to retrieve the proper motor settings, you need to provide
        the stage name.

        Args:
            serial (str, optional): Serial number. Defaults to None.
            stage_name (str, optional): Stage name (written on the stage). Defaults to 'Z825'.

        Returns:
            GenericDevice: The KDC101 object.
        """
        self.device_prefix = KCubeDCServo.DevicePrefix
        super().__init__(serial=serial, device_prefix=self.device_prefix)
        self.attempt_connection()
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

    def attempt_connection(self) -> None:
        """Attempt connection.

        Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        """
        try:
            self.device = KCubeDCServo.CreateKCubeDCServo(self.serial)
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
                "Success ! Connected to KCube motor"
                + f" {self.device_info.SerialNumber}"
                + f" {self.device_info.Name}"
            )
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
