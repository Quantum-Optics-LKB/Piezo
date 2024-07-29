from pylablib.devices import Thorlabs

conn = {
    "port": "/dev/ttyUSB0",
    "baudrate": 115200,
    "rtscts": True,
}  # intead of ttyUSB0 use the correct path
dev = Thorlabs.KinesisMotor(("serial", conn))
dev.close()
