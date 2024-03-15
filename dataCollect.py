from dataConvert import *
import scipy.io as sio
import pathlib as pl
import traceback
import cv2
import os
import shutil

SCENETYPE = ['a', 'b']

class DataCollect:
    def __init__(self) -> None:
        self.IR_imgs = []
        self.heat_imgs = []
        self.recording = False

        self.root = pl.Path('./data')

    def key_handler(self, key):
        if key == 32:
            # space
            self.recording = not self.recording
            print(f'recording now {self.recording}')

            if not self.recording and len(self.IR_imgs) > 0:
                self.save_data()
        elif key == 27:
            # esc, to re-save the last round file to another directory
            self.resave_data()

    def resave_data(self):
        new_scenetype = self._scenetype
        new_sceneroot = self.root / pl.Path(new_scenetype)
        os.makedirs(new_sceneroot, exist_ok=True)
        try:
            for ext in ['.mat', '.mp4']:
                shutil.move(f'{self.last_sceneroot / self.last_filename}{ext}', 
                            f'{new_sceneroot / self.last_filename}{ext}')
            print("moving file completed!")

            # save some parameters to instance
            self.last_sceneroot = new_sceneroot
            self.last_scenetype = new_scenetype
        
            # another change for re-saving the files
            print("上一轮的存储是否想改变主意？按下ESC以重新保存，否则请忽略。")

        except Exception as e:
            traceback.print_exc()


    def save_data(self):
        print("Saving data...")
        

        while True:
            try:
                scenetype = self._scenetype
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
                self.last_sceneroot = sceneroot
                self.last_scenetype = scenetype
                self.last_filename = filename

                # another change for re-saving the files
                print("上一轮的存储是否想改变主意？按下ESC以重新保存，否则请忽略。")
                
                break
            except Exception as e:
                traceback.print_exc()
    

    @property
    def _scenetype(self):
        print(list(zip(range(1, len(SCENETYPE)+1), SCENETYPE)))
        labels = int(input('Specify a scenetype from above ->'))
        scenetype = SCENETYPE[labels-1]

        return scenetype



    def pre_zoom(self, IR_img):
        if self.recording:
            self.IR_imgs.append(IR_img)

    def after_zoom(self, heat_img):
        if self.recording:
            self.heat_imgs.append(heat_img)


if __name__ == '__main__':
    my_data_collector = DataCollect()
    play_IR(my_data_collector.pre_zoom, my_data_collector.after_zoom, my_data_collector.key_handler)

