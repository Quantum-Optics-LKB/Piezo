# -*- coding: utf-8 -*-
import clr
import sys
import time
import traceback
from .GenericDevice import GenericDevice

# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
clr.AddReference("Thorlabs.MotionControl.Controls")

from System import Decimal
import System.Collections
from System.Collections import *
# Generic device manager
import Thorlabs.MotionControl.Controls

from Thorlabs.MotionControl.GenericMotorCLI import *

from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *

class LTS(GenericDevice):
   
    def __init__(self, serial: str = None) -> GenericDevice:
        """Instantiates a K10CR1 object to control cage rotator

        :param str serial: Serial number
        :return: K10CR1 object

        """
        self.device_prefix = LongTravelStage.DevicePrefix
        super().__init__(serial=serial, device_prefix=self.device_prefix)
        self.attempt_connection()
        self.configuration = self.device.LoadMotorConfiguration(self.serial)
        self.settings = self.device.MotorDeviceSettings
        # Sets the velocities and acceleration to their maximum values
        velParams = self.device.GetVelocityParams()
        velParams.Acceleration = Decimal(10)
        velParams.MaxVelocity = Decimal(15)
        velParams.MinVelocity = Decimal(0)
        self.device.SetVelocityParams(velParams)

    def attempt_connection(self) -> None:
        """Attempt connection.
        
        Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        """
        try:
            self.device = LongTravelStage.CreateLongTravelStage(self.serial)
            self.device.Connect(self.serial)
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