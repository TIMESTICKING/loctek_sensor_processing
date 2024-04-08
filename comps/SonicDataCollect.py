import csv
import shutil
import struct
import traceback
from .utils import *
from PyQt6.QtCore import QObject,pyqtSignal

class SonicDataCollect(QObject):

    new_dist_signal = pyqtSignal(float)
    def __init__(self, queue, socket=None, name='sonic1') -> None:
        super().__init__()
        self.socket = socket
        self.queue = queue
        self.name = name
        self.distances = []
        self.pickinterval = 0

    def play_sonic(self):
        while True:
            sonic_raw = self.queue.get() # wait for an avaliable item

            # 将字节数组转换为浮点数列表
            try:
                float_number = struct.unpack('f', sonic_raw)[0]
                if CONTROL.RECORDING or CONTROL.TESTING:
                    self.pickinterval = self.pickinterval - 1
                    if self.pickinterval <= 0:
                        self.distances.append(float_number)
                        print('超声增加1帧')
                        self.pickinterval = CONTROL.Sonic_interval
                if self.socket is not None and self.socket[1] is not None:
                    self.socket[1].sendall(str(float_number).encode())
                else:
                    self.new_dist_signal.emit(float_number)

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

