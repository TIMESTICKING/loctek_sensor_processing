from dataConvert import *
import scipy.io as sio
import pathlib as pl
import traceback
import cv2
import os

SCENETYPE = ['a', 'b']

class DataCollect:
    def __init__(self) -> None:
        self.IR_imgs = []
        self.heat_imgs = []
        self.recording = False

    def key_handler(self, key):
        if key == 32:
            # space
            self.recording = not self.recording
            print(f'recording now {self.recording}')

            if not self.recording and len(self.IR_imgs) > 0:
                self.save_data()


    def save_data(self):
        print("Saving data...")
        root = pl.Path('./data')

        print(list(zip(range(1, len(SCENETYPE)+1), SCENETYPE)))
        while True:
            try:
                labels = int(input('Specify a scenetype from above ->'))
                scenetype = SCENETYPE[labels-1]
                sceneroot = root / pl.Path(scenetype)
                timestamp = int(time.time())
                filename = f'{scenetype}_{timestamp}'
                os.makedirs(sceneroot, exist_ok=True)

                # save IR_img np.array as mat
                sio.savemat(sceneroot / filename,
                            {'IR_video': np.array(self.IR_imgs)})
                
                # save preview heatmap videos
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
                video_writer = cv2.VideoWriter(f'{sceneroot / filename}.mp4', fourcc, 10.0, (160, 160))
                for hmap in self.heat_imgs:
                    video_writer.write(hmap) 
                video_writer.release()
                
                break
            except Exception as e:
                traceback.print_exc()
    

            

    def pre_zoom(self, IR_img):
        if self.recording:
            self.IR_imgs.append(IR_img)

    def after_zoom(self, heat_img):
        if self.recording:
            self.heat_imgs.append(heat_img)


if __name__ == '__main__':
    my_data_collector = DataCollect()
    play_IR(my_data_collector.pre_zoom, my_data_collector.after_zoom, my_data_collector.key_handler)

