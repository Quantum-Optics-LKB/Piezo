# Piezo
**Python interface for Thorlabs piezo devices** 

Introduction
===============================================

This little library intends to map the Kinesis .NET API to python in order to control various Thorlabs piezos, motors etc ... The goal here is to bring the relevant Kinesis functions to practical high level classes. Each device has its own class, for instance a typical piezo actuated screw is controlled by a TCube controller, and corresponds to a Thorlabs TIM101 TCube Inertial Motor object. 

Communication through Kinesis
==========================

Communication with the devices is achieved by using Kinesis DLL's through the Kinesis .NET API. This API is then accessed through a direct one to one mapping to python achieved thanks to pythonnet. The `clr` module of pythonnet allows to add the references of the objects we want to use as can be seen at the top of `piezo.py`. The references are added as follows :
```python
import clr
import sys

sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")

# Add references so Python can see .Net
clr.AddReference("System")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
```
In the above example, we have added the `System` .NET namespace as well as the `Thorlabs.MotionControl.DeviceManagerCLI` namespace. The latter is the utility handling detection of the devices. After adding a reference, the namespace can be accessed as a regular Python class. One can then import all of its attributes and methods as follows :

```python
from System import String
from System import Decimal
import System.Collections
from System.Collections import *
# Generic device manager
from Thorlabs.MotionControl.DeviceManagerCLI import *
```

Once imported, all of the methods behave just like their .NET counterparts, but as Python objects. Function signatures are converted to the matching Python type by using the `System` types such as `String` or `Decimal`.

Adding devices 
==============

Adding a new device to the library is a fairly straightforward operation. For this, one needs to take a look at the Kinesis .NET API documentation provided in the Kinesis folder. The first step is to find the Thorlabs model name of the device we wish to connect. This is easily done by opening the Kinesis app. Then, one needs to go in the documentation at the relevant chapter. 

In the contents table on the ![left pan](images/leftpane.png) of the compiled HTML viewer, you will find the list of all Thorlabs devices supported by the Kinesis API. 

Dependencies 
============
* Pythonnet : Mapping of the .NET functions to Python. **WARNING** Pythonnet provides the `clr` module. If you attempt to `pip install` it you will end up with the `color` module which will be nice, but not what is needed here.
* For Unix systems : `mono` which is the open implementation of .NET. It can be installed directly through apt : `sudo apt install mono-complete`
* Numpy
* Kinesis : This code directly uses Kinesis DLL's. It can be run under all OS provided the path to the Kinsesis folder is added at the begining of `piezo.py` (l.24)
