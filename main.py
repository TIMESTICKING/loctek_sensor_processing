import struct
from drivers.Serial import *
import numpy as np
import cv2
from scipy.ndimage import zoom

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


def play_IR():
    for IR_float in convert2float():
        IR_img = np.array(IR_float).reshape(8, 8)
        # print(IR_img)
        # IR_img = IR_img / IR_img.max()

        zoomed_IR = zoom(IR_img, (20, 20), order=3).clip(0, 40)
        zoomed_IR_int = np.asarray(zoomed_IR * 6.375, np.uint8)
        heatmap = cv2.applyColorMap(zoomed_IR_int, cv2.COLORMAP_JET)

        cv2.imshow('IR_img', heatmap)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

play_IR()
