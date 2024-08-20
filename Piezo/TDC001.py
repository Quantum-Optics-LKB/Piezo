# -*- coding: utf-8 -*-

import time
import traceback

import clr

from .GenericDevice import GenericDevice

# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
clr.AddReference("Thorlabs.MotionControl.Controls")


# Generic device manager
from System import Decimal
from System.Collections import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *
from Thorlabs.MotionControl.TCube.DCServoCLI import *
from Thorlabs.MotionControl.TCube.InertialMotorCLI import *


class TDC001(GenericDevice):
    def __init__(self, serial: str = None) -> GenericDevice:
        """Instantiates a TDC001 object to control piezo DC motor actuators

        :param str serial: Serial number
        :return: TDC001 object

        """
        self.device_prefix = TCubeDCServo.DevicePrefix
        super().__init__(serial, self.device_prefix)
        self.attempt_connection()
        self.configuration = self.device.LoadMotorConfiguration(self.serial)
        self.settings = self.device.MotorDeviceSettings
        msg = f"{self.short_name} | "
        msg += f"Configured for {self.configuration.DeviceSettingsName} stage"
        print(msg)
        # do we home the device upon initialization ?
        # for task completion
        self.__taskID = 0
        self.__taskComplete = False

    def attempt_connection(self):
        """Attempt connection to the device.

        Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        """
        try:
            self.device = TCubeDCServo.CreateTCubeDCServo(self.serial)
            self.device.Connect(self.serial)
            timeout = 0
            while not (self.device.IsSettingsInitialized()) and (timeout <= 10):
                self.device.WaitForSettingsInitialized(500)
                timeout += 1
            self.device.StartPolling(250)
            time.sleep(0.5)
            self.device.EnableDevice()
            self.device_info = self.device.GetDeviceInfo()
            self.short_name = self.device_info.Name
            print(
                "Success ! Connected to TCube motor:"
                + f" {self.device_info.Description}"
                + f", S/N: {self.device_info.SerialNumber}"
            )
        except Exception:
            print("ERROR : Could not connect to the device")
            print(traceback.format_exc())

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
        #     sys.stdout.write(f"\rHoming device..Position : {status.Position}")
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
        return float(str(self.device.Position))
    
    def get_jog_stepsize(self):
        return float(str(self.device.GetJogStepSize()))
    
    def set_jog_stepsize(self, stepsize):
        self.device.SetJogStepSize(Decimal(stepsize))
        params = self.device.GetJogParams()
        return float(params.StepSize.ToString())
    
    def jog(self, direction: int = 1):
        if direction == 1:
            self.device.MoveJog(MotorDirection.Forward, None)
        elif direction == -1:
            self.device.MoveJog(MotorDirection.Forward, None)

