import struct


class SonicDataCollect:

    def __init__(self, queue, name='sonic1') -> None:
        self.queue = queue
        self.name = name

    def play_sonic(self):
        while True:
            sonic_raw = self.queue.get() # wait for an avaliable item

            # 将字节数组转换为浮点数列表
            try:
                float_number = struct.unpack('f', sonic_raw)[0]
            except Exception as e:
                pass
            print(f'{self.name} is {float_number}')
