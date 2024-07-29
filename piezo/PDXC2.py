# -*- coding: utf-8 -*-

import clr
import time
import traceback
from .GenericDevice import GenericDevice

# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.Controls")

import System.Collections
from System.Collections import *
# Generic device manager
import Thorlabs.MotionControl.Controls

from Thorlabs.MotionControl.Benchtop.Piezo import *
from Thorlabs.MotionControl.Benchtop.PiezoCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI import *
from Thorlabs.MotionControl.Benchtop.PiezoCLI.PDXC2 import *
from Thorlabs.MotionControl.Benchtop.Piezo.PDXC2 import *
from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes

class PDXC2(GenericDevice):

    def __init__(self, serial: str = None, mode: int = 2) -> GenericDevice:
        """Instantiates a PDXC2 object to control piezo stage.

        :param str serial: Serial number
        :param int mode: OpenLoop (1) or CloseLoop (2)
        :return: PDXC2 object

        """
        self.device_prefix = InertiaStageController.DevicePrefix_PDXC2
        super().__init__(serial=serial, device_prefix=self.device_prefix)
        self.attempt_connection(mode)  
        self.position = self.get_position()
        self.control_mode = self.get_position_control_mode()
        
    def attempt_connection(self, mode: int) -> None:
        """Attempt connection.

        Generic connection attempt method. Will try to connect to specified
        serial number after device lists have been built. Starts all relevant
        routines as polling / command listeners ...

        Args:
            mode (int): Open or closed loop control mode. 1 for OpenLoop, 2 for CloseLoop.

        """
        try:
            self.device = InertiaStageController.CreateInertiaStageController(self.serial)
            self.device.Connect(self.serial)
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

            self.configuration = self.device.GetPDXC2Configuration(self.serial, DeviceConfiguration.DeviceSettingsUseOptionType.UseDeviceSettings)
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
        print(f'Device {self.device_info[0].SerialNumber} disconnected.')