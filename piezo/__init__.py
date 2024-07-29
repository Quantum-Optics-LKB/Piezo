"""
Piezo.

A package to control Thorlabs devices using the Kinesis .NET API.
"""

__version__ = "0.0.1"
__author__ = "Tangui Aladjidi; Lucien Belzane"
__license__ = "GPLv3"
__credits__ = "Laboratoire Kastler Brossel, Paris, France"
__email__ = "tangui.aladjidi@lkb.upmc.fr"
#TODO Find out if we can do the .NET imports here ? how does it pass down to
# the inherited classes ? 
from .GenericDevice import *
from .PDXC2 import *
from .K10CR1 import *
from .TDC001 import *
from .TIM101 import *
from .BPC import *
from .KIM101 import *
from .KDC101 import *
from .LTS import *