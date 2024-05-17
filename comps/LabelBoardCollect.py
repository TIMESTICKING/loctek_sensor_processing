import csv
import shutil
import struct
import traceback

import numpy as np
from .utils import *
from PyQt5.QtCore import QObject,pyqtSignal
from comps.IRdataCollect import IR_byte_decoder


class LabelBoardCollect:
    def __init__(self) -> None:
        pass

    def play(self):
        while True:
            label_raw = MESSAGE.label_board.get() # wait for an avaliable item

            # 将字节数组转换为浮点数列表
            try:
                label_by_board = label_raw[0]
                label = IR_byte_decoder(label_raw[1:])

                ind = np.argmax(label)

                print(label_by_board)

            except Exception as e:
                traceback.print_exc()
