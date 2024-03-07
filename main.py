import struct
from drivers.Serial import *


serial = MySerial_2head1tail(b'\x88', 'COM3', b'\x77', b'\x66', 64 * 4)

def convert2float():
    for res in serial.readData():
        # print(res)

        # 将字节数组转换为浮点数列表
        IR_float = []
        for i in range(0, len(res), 4):
            float_number = struct.unpack('f', res[i:i+4])[0]
            IR_float.append(float_number)

        yield IR_float
