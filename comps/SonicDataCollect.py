import csv
import shutil
import struct
import traceback
from .utils import *

class SonicDataCollect:

    def __init__(self, queue, socket=None, name='sonic1') -> None:
        self.socket = socket
        self.queue = queue
        self.name = name
        self.distances = []

    def play_sonic(self):
        while True:
            sonic_raw = self.queue.get() # wait for an avaliable item

            # 将字节数组转换为浮点数列表
            try:
                float_number = struct.unpack('f', sonic_raw)[0]
                if CONTROL.RECORDING:
                    self.distances.append(float_number)
                
                if self.socket is not None and self.socket[1] is not None:
                    self.socket[1].sendall(str(float_number).encode())
                else:
                    print(f'{self.name} is {float_number}')
            except Exception as e:
                traceback.print_exc()


    def resave_data(self, new_scenetype, new_filename, new_sceneroot):
        try:
            shutil.move(f'{CONTROL.last_sceneroot / CONTROL.last_filename}.csv', 
                            f'{new_sceneroot / new_filename}.csv')
        except Exception as e:
            traceback.print_exc()


    def save_data(self, scenetype, filename, sceneroot):
        print(f"Saving {self.name} data...")

        with open(f'{sceneroot / filename}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.distances)
        
        self.clear_buffer()


    def clear_buffer(self):
        self.distances = []

