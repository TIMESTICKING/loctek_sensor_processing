"""
@Description :   
@Author      :   李佳宝
@Time        :   2024/04/24 11:03:33
"""

import scipy.io as sio
import pathlib as pl
import traceback
import cv2
from scipy.ndimage import zoom
import os
import shutil
import struct
import numpy as np
import time
from .utils import MESSAGE, CONTROL
import threading
from PyQt5.QtCore import QObject, pyqtSignal

def IR_byte_decoder(IR_raw):
    IR_float = []
    for i in range(0, len(IR_raw), 4):
        float_number = struct.unpack('f', IR_raw[i:i+4])[0]
        IR_float.append(float_number)
    IR_img = np.array(IR_float)

    return IR_img


class IRDataCollect(QObject):
    new_heatmap_signal = pyqtSignal(np.ndarray)


    def __init__(self) -> None:
        super().__init__()
        self.IR_imgs = []
        self.heat_imgs = []
        self.trig_stop = False
    def resave_data(self, new_scenetype, new_filename, new_sceneroot):
        try:
            for ext in ['.mat', '.mp4']:
                shutil.move(f'{CONTROL.last_sceneroot / CONTROL.last_filename}{ext}', 
                            f'{new_sceneroot / new_filename}{ext}')
            print("moving file completed!")

        except Exception as e:
            traceback.print_exc()


    def save_data(self, scenetype, filename, sceneroot):
        print("Saving IR data...")
        
        # save IR_img np.array as mat
        sio.savemat(f'{sceneroot / filename}.mat',
                    {'IR_video': np.array(self.IR_imgs)}, appendmat=True)
        
        # save preview heatmap videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
        video_writer = cv2.VideoWriter(f'{sceneroot / filename}.mp4', fourcc, 10.0, (160, 160))      
        for hmap in self.heat_imgs:
            video_writer.write(hmap) 
        video_writer.release()
        
        # self.clear_buffer()                
  

    def clear_buffer(self):
        # clear buffer
        self.heat_imgs = []
        self.IR_imgs = []

    def stop_thread(self):
        self.trig_stop = True

    def play_IR(self):
        while not self.trig_stop:
            IR_raw = MESSAGE.IR.get() # wait for an avaliable item

            # 将字节数组转换为浮点数列表
            IR_img = IR_byte_decoder(IR_raw)
            MESSAGE.IR_net_ready.append(IR_img)
            IR_img = IR_img.reshape(8, 8)

            self._pre_zoom(IR_img) # hook function
            # print(IR_img)
            # IR_img = IR_img / IR_img.max()

            zoomed_IR = zoom(IR_img, (20, 20), order=3).clip(15, 40)
            zoomed_IR_int = np.asarray((zoomed_IR - 15) * (255 / 25), np.uint8)
            heatmap = cv2.applyColorMap(zoomed_IR_int, cv2.COLORMAP_JET)

            self._after_zoom(heatmap)   # hook function
            self.new_heatmap_signal.emit(heatmap)

            #  cv2.imshow('IR_img', heatmap)
            #  key = cv2.waitKey(1)
            #  if key != -1:
            #     MESSAGE.KEY.put(key, timeout=1)

        # raise Exception('System terminated by user at "q"')



    def _pre_zoom(self, IR_img):
        if CONTROL.RECORDING or CONTROL.TESTING:            
            self.IR_imgs.append(IR_img)

    def _after_zoom(self, heat_img):
        if CONTROL.RECORDING or CONTROL.TESTING:
            self.heat_imgs.append(heat_img)


if __name__ == '__main__':
    my_data_collector = IRDataCollect()
    my_data_collector.play_IR()

