from queue import Queue

class MESSAGE:

    IR:Queue = Queue(5)
    sonic1:Queue = Queue(5)


class TAG:
    SONIC1 = 0x02
    IR = 0x01

