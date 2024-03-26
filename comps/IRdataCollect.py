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

SCENETYPE = ['sitting', 'standing']

class IRDataCollect:
    def __init__(self) -> None:
        self.IR_imgs = []
        self.heat_imgs = []
        CONTROL.RECORDING = False

        self.root = pl.Path('./data')


    def async_record_or_pause(self):
        CONTROL.RECORDING = not CONTROL.RECORDING
        print(f'recording now {CONTROL.RECORDING}')

        if not CONTROL.RECORDING and len(self.IR_imgs) > 0:
            threading.Thread(target=self._save_data).start()


    def async_resave_data(self):
        threading.Thread(target=self._resave_data).start()



    def _resave_data(self):
        new_scenetype = self._scenetype()
        new_sceneroot = self.root / pl.Path(new_scenetype)
        os.makedirs(new_sceneroot, exist_ok=True)
        try:
            for ext in ['.mat', '.mp4']:
                shutil.move(f'{CONTROL.last_sceneroot / CONTROL.last_scenetype}{ext}', 
                            f'{new_sceneroot / CONTROL.last_scenetype}{ext}')
            print("moving file completed!")

            # save some parameters to instance
            CONTROL.last_sceneroot = new_sceneroot
            CONTROL.last_scenetype = new_scenetype
        
            # another change for re-saving the files
            print("上一轮的存储是否想改变主意？按下ESC以重新保存，否则请忽略。")

        except Exception as e:
            traceback.print_exc()


    def _save_data(self):
        print("Saving data...")
        
        scenetype = self._scenetype()
        sceneroot = self.root / pl.Path(scenetype)
        timestamp = int(time.time())
        filename = f'{scenetype}_{timestamp}'
        os.makedirs(sceneroot, exist_ok=True)

        # save IR_img np.array as mat
        sio.savemat(f'{sceneroot / filename}.mat',
                    {'IR_video': np.array(self.IR_imgs)}, appendmat=True)
        
        # save preview heatmap videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
        video_writer = cv2.VideoWriter(f'{sceneroot / filename}.mp4', fourcc, 10.0, (160, 160))
        for hmap in self.heat_imgs:
            video_writer.write(hmap) 
        video_writer.release()

        # clear buffer
        self.heat_imgs = []
        self.IR_imgs = []

        # save some parameters to instance
        CONTROL.last_sceneroot = sceneroot
        CONTROL.last_scenetype = scenetype
        CONTROL.last_scenetype = filename

        # another change for re-saving the files
        print("上一轮的存储是否想改变主意？按下ESC以重新保存，否则请忽略。")
                
  

    def _scenetype(self):
        print(list(zip(range(1, len(SCENETYPE)+1), SCENETYPE)))
        labels = int(input('Specify a scenetype from above, 0 to discard ->'))
        if labels == 0:
            # clear buffer
            self.heat_imgs = []
            self.IR_imgs = []
            raise Exception('You have discarded the data, now continue...')
        
        assert 0 <= labels-1 < len(SCENETYPE), 'label ID out of range!'
        scenetype = SCENETYPE[labels-1]

        return scenetype


    def play_IR(self):
        while True:
            IR_raw = MESSAGE.IR.get() # wait for an avaliable item

            # 将字节数组转换为浮点数列表
            IR_float = []
            for i in range(0, len(IR_raw), 4):
                float_number = struct.unpack('f', IR_raw[i:i+4])[0]
                IR_float.append(float_number)
            IR_img = np.array(IR_float).reshape(8, 8)


            self._pre_zoom(IR_img) # hook function
            # print(IR_img)
            # IR_img = IR_img / IR_img.max()

            zoomed_IR = zoom(IR_img, (20, 20), order=3).clip(15, 40)
            zoomed_IR_int = np.asarray((zoomed_IR - 15) * (255 / 25), np.uint8)
            heatmap = cv2.applyColorMap(zoomed_IR_int, cv2.COLORMAP_JET)

            self._after_zoom(heatmap) # hook function

            cv2.imshow('IR_img', heatmap)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            else:
                MESSAGE.KEY.put(key, timeout=1)
        
        # raise Exception('System terminated by user at "q"')



    def _pre_zoom(self, IR_img):
        if CONTROL.RECORDING:
            self.IR_imgs.append(IR_img)

    def _after_zoom(self, heat_img):
        if CONTROL.RECORDING:
            self.heat_imgs.append(heat_img)


if __name__ == '__main__':
    my_data_collector = IRDataCollect()
    my_data_collector.play_IR()

