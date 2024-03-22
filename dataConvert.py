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

        IR_img = np.array(IR_float).reshape(8, 8)
        yield IR_img


def play_IR(fun_prezoom=lambda x: None, fun_afterzoom=lambda x: None, key_handler=lambda x: None):
    for IR_img in convert2float():
        fun_prezoom(IR_img) # hook function
        # print(IR_img)
        # IR_img = IR_img / IR_img.max()

        zoomed_IR = zoom(IR_img, (20, 20), order=3).clip(15, 40)
        zoomed_IR_int = np.asarray((zoomed_IR - 15) * (255 / 25), np.uint8)
        heatmap = cv2.applyColorMap(zoomed_IR_int, cv2.COLORMAP_JET)

        fun_afterzoom(heatmap) # hook function

        cv2.imshow('IR_img', heatmap)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        else:
            key_handler(key)







if __name__ == '__main__':
    play_IR()
