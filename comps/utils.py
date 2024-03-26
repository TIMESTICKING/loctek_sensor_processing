from queue import Queue


class MESSAGE:
    KEY = Queue(5)
    IR:Queue = Queue(5)
    sonic1:Queue = Queue(5)


class CONTROL:
    RECORDING = False
    last_sceneroot = ''
    last_scenetype = ''
    last_filename = ''



class TAG:
    SONIC1 = 0x02
    IR = 0x01

