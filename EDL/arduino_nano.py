
import os
import serial

import serial.tools.list_ports
print(list(serial.tools.list_ports.comports()))
ser= serial.Serial('/dev/ttyACM0', 115200)
ser.write(b'Start')

line = ser.readline()
print(line)
ser.close()

