import struct
from .utils import *

class SonicDataCollect:

    def __init__(self, queue, name='sonic1') -> None:
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
                print(f'{self.name} is {float_number}')
            except Exception as e:
                pass


    def save_data(self, scenetype, filename, sceneroot):
        print(f"Saving {self.name} data...")



