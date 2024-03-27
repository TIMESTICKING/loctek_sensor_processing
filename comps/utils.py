import os
from queue import Queue
import pathlib as pl
import time


class MESSAGE:
    KEY = Queue(5)
    IR:Queue = Queue(5)
    sonic1:Queue = Queue(5)
    scene_type = Queue(5)


class CONTROL:
    SCENETYPES = ['sitting', 'standing']
    ROOT = pl.Path('./data')

    RECORDING = False
    last_sceneroot = ''
    last_scenetype = '' 
    last_filename = ''

    @staticmethod
    def get_scenetype():
        print(list(zip(range(1, len(CONTROL.SCENETYPES)+1), CONTROL.SCENETYPES)))
        labels = int(input('Specify a scenetype from above, 0 to discard ->'))
        if labels == 0:
            print('You have discarded the data, now continue...')
            return False
        
        assert 0 <= labels-1 < len(CONTROL.SCENETYPES), 'label ID out of range!'
        scenetype = CONTROL.SCENETYPES[labels-1]
        sceneroot = CONTROL.ROOT / pl.Path(scenetype)
        timestamp = int(time.time())
        filename = f'{scenetype}_{timestamp}'
        os.makedirs(sceneroot, exist_ok=True)


        return [scenetype, filename, sceneroot]

    @staticmethod
    def update_lastround(args):
        # save some parameters to instance
        CONTROL.last_sceneroot = args[2]
        CONTROL.last_scenetype = args[0]
        CONTROL.last_filename = args[1]


class TAG:
    SONIC1 = 0x02
    IR = 0x01

