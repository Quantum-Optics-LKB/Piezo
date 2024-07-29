# -*- coding: utf-8 -*-

import clr
import sys
import time
import traceback
from .GenericDevice import GenericDevice

# Add references so Python can see .Net
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
clr.AddReference("Thorlabs.MotionControl.Controls")

from System import Decimal
import System.Collections
from System.Collections import *
import Thorlabs.MotionControl.Controls

from Thorlabs.MotionControl.GenericPiezoCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *

from Thorlabs.MotionControl.KCube.InertialMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *

from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *

class K10CR1(GenericDevice):
    def __init__(self, serial: str = None) -> object:
        """Instantiate a K10CR1 object.

        Args:
            serial (str, optional): Serial number. Defaults to None.

        Returns:
            object: The K10CR1 object.
        """
        self.device_prefix = CageRotator.DevicePrefix
        super().__init__(serial, self.device_prefix)
        self.attempt_connection()
        self.configuration = self.device.LoadMotorConfiguration(self.serial)
        self.settings = self.device.MotorDeviceSettings
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
            self.device = CageRotator.CreateCageRotator(self.serial)
            self.device.Connect(self.serial)
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