from datetime import datetime
from ctypes import cdll,c_long, c_ulong, c_uint32,byref,create_string_buffer,c_bool,c_char_p,c_int,c_int16,c_double, sizeof, c_voidp
from TLPMX import TLPMX
import time

from TLPMX import TLPM_DEFAULT_CHANNEL

class PMXXX():
    """Instantiates a PMXXX object to control thorlabs powermeter.
    Tested with PM400.
    :return: PMXXX object

    """
    def __init__(self) -> None:
        self.tlPM = TLPMX()
        self.deviceCount = None
        self.resourceName = None
        self.message = None
        self.wavelength = None
        self.attempt_connection()

    def attempt_connection(self):
        ## TODO: Choose device with serial number. 
        self.deviceCount = c_uint32()
        self.tlPM.findRsrc(byref(self.deviceCount))
        print("Number of found devices: " + str(self.deviceCount.value))
        print("")  
        self.resourceName = create_string_buffer(1024)
        self.tlPM.getRsrcName(c_int(0), self.resourceName)

        self.tlPM.open(self.resourceName, c_bool(True), c_bool(True))

        self.message = create_string_buffer(1024)
        self.tlPM.getCalibrationMsg(self.message,TLPM_DEFAULT_CHANNEL)
        print("Connected to device", 1)
        print("Last calibration date: ",c_char_p(self.message.raw).value)
        print("")

    def set_wavelength(self, value: float = 532.0):
        """
        Set wavelength in nm.
        """
        self.wavelength = c_double(value)
        self.tlPM.setWavelength(self.wavelength,TLPM_DEFAULT_CHANNEL)
        
    def set_power_unit(self, value: float = 0):
        """
        Set power unit to Watt.
        0 -> Watt
        1 -> dBm
        """
        self.tlPM.setPowerUnit(c_int16(value),TLPM_DEFAULT_CHANNEL)

    def set_autorange(self, value: bool):
        """
        Enable auto-range mode.
        """
        if value:
            self.tlPM.setPowerAutoRange(c_int16(1),TLPM_DEFAULT_CHANNEL)
        else:
            self.tlPM.setPowerAutoRange(c_int16(0),TLPM_DEFAULT_CHANNEL)

    def get_power(self):
        """
        Get power in W or dBm.
        """
        power =  c_double()
        self.tlPM.measPower(byref(power),TLPM_DEFAULT_CHANNEL)
        return float(power.value)

    def disconnect(self):
        """
        Disconnect power meter.
        """
        
        self.tlPM.close()
