import queue
from time import sleep
import numpy as np
import random
import serial.tools.list_ports
import sys
from drivers.Serial import *
from argparse import ArgumentParser
from comps.utils import *
from comps.IRdataCollect import *
from comps.SonicDataCollect import *
from comps.LabelBoardCollect import *
import threading

class DEVICE:
    IR_data_collector = IRDataCollect()
    board_collector = LabelBoardCollect()
    sonic_device1 = SonicDataCollect(MESSAGE.sonic1, SOCKET.sonic1, 'sonic1')

class IRCaliFifo:
    def __init__(self, max_size:int = 0):
        self.t_k = threading.Lock()
        self._max_size = max_size
        self._raw = []
        self._averages = []

    def putFrame(self, frame:list):
        with self.t_k:
            if self._max_size > 0 and len(self._raw)>= self._max_size:
                self._raw.pop(0)
                self._averages.pop(0)
            self._raw.append(frame)
            self._averages.append(np.average(frame))

    def length(self) -> int:
        with self.t_k:
            return len(self._raw)

    def getAvg(self) -> list:
        with self.t_k:
            return list(self._averages)
    
    def calc(self)->list:
        with self.t_k:
            return [np.average(self._averages),np.max(self._averages),np.min(self._averages) ]
    
    def calcToalAvg(self) -> float:
        with self.t_k:
            return np.average(self._averages)
    def calcMaxAvg(self) -> float:
        with self.t_k:
            return np.max(self._averages)  
    def calcMinAvg(self) -> float:
        with self.t_k:
            return np.min(self._averages)  
    
    def isFull(self) -> bool:
        with self.t_k:
            if self._max_size <= 0:
                return False
            return len(self._averages)>= self._max_size

class MySerial_2head1tail(MySerial):
    def __init__(self,fifo:IRCaliFifo, h2,*args, **kwargs):
        super(MySerial_2head1tail, self).__init__(*args, **kwargs)
        # self.frame_len = frame_len
        self.h2 = h2
        self.__fifo = fifo
        self.trig_stop = False

    def stop_thread(self):
        self.trig_stop = True

    def readData(self):
        buf = b''
        sta = SM2h2t.findinghead1
        read_cnt = 0
        max_read = 64 * 10
        while not self.trig_stop:
            read_cnt += 1
            data = self.port.read()
            # print(data.hex())
            # if len(data) == 0 or read_cnt > max_read:
            #     warnings.warn('串口读取超时')
            #     self.portClose()
            #     break

            if sta == SM2h2t.findinghead1:
                if data == self.h:
                    # buf += data
                    sta = SM2h2t.findinghead2
            elif sta == SM2h2t.findinghead2:
                if data == self.h2:
                    # buf += data
                    sta = SM2h2t.findingtail
            elif sta == SM2h2t.findingtail:
                if data == self.t and (len(buf) in self.length or self.length is None):
                    yield buf
                    read_cnt = 0
                    buf = b''
                    sta = SM2h2t.findinghead1
                else:
                    buf += data

    def writeData(self, data):
        self.port.write((data + '\n').encode('utf-8'))

    
    def start_report(self):
        time.sleep(2)
        self.writeData("{'cmd':'SAC','debug': false ,'spit': true , 'model': 0,'table_ctrl':false,'reconnect':false}")


    def message_classify(self):
        for res in self.readData():
            # print(res)

            paras = (res[1:], True, None)
            try:
                if res[0] == TAG.IR:
                    IR_array = IR_byte_decoder(res[1:])
                    self.__fifo.putFrame(IR_array)
                    # IR_img = IR_img.reshape(8, 8)
                    # print("IR_img:")
                    # print(IR_img)
                    # print("\r\n")
                # if res[0] == TAG.IR and not MESSAGE.IR.full():
                    # MESSAGE.IR.put(*paras)
                # elif res[0] == TAG.SONIC1 and not MESSAGE.sonic1.full():
                #     MESSAGE.sonic1.put(*paras)
                # elif res[0] == TAG.LABEL and not MESSAGE.label_board.full():
                #     MESSAGE.label_board.put(*paras)
            except Exception as e:
                traceback.print_exc()
                break


if __name__ == "__main__":
    ir_fifo = IRCaliFifo(120)
    myserial = MySerial_2head1tail(ir_fifo,b'\xFA',"COM12", b'\xAF', b'\xFF', length=[64 * 4 + 1, 5, 22])
    workThreadsList = []
    jobs = [myserial.message_classify]
    for job in jobs:
        t = threading.Thread(target=job, )
        t.daemon = True
        t.start()
        workThreadsList.append(t)

    try:
        myserial.start_report()
        print("检测连接成功:")
        # # 连接升降桌
        # self.tableController.startCtrl(port=self.com_table)
        # print("升降桌连接成功:")
        # print("升降桌串口:", self.com_table)
    except Exception as e:
        print("设备串口连接失败！")

    
    while not (ir_fifo.isFull()):
        v_avg,v_max,v_min = ir_fifo.calc()
        print(f'len:{ir_fifo.length()} avg:{v_avg} max:{v_max} min:{v_min}')
        sleep(0.5)

    v_avg,v_max,v_min = ir_fifo.calc()
    print(f'len:{ir_fifo.length()} avg:{v_avg} max:{v_max} min:{v_min}')


    myserial.stop_thread()
        