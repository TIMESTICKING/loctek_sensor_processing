# -*- coding: utf-8 -*-
# @Time    : 2021/10/29 22:22
# @Author  : Jiabao Li
# @FileName: Serial.py
# @Software: PyCharm
import warnings

import serial
# import serial.tools.list_ports
import threading
import time
import traceback


class SM:
    findinghead=0
    findingtail=1

class MySerial:

    def __init__(self, port, h=b'\x55', t=b'\xaa', length=None, buandRate=115000, timeout=2):
        self.length = length
        self.t = t
        self.h = h

        self.port = serial.Serial(port, buandRate, timeout=timeout)
        if not self.port.isOpen():
            self.port.open()
        # self.start_listen()

    def portOpen(self):
        if not self.port.isOpen():
            self.port.open()

    def portClose(self):
        self.port.close()

    def sendData(self, data):
        # if not self.port.isOpen():
        #     self.port.open()

        number = self.port.write(data)
        return number

    def readData(self):
        buf = b''
        sta = SM.findinghead
        read_cnt = 0
        max_read = 200
        while True:
            # try:
                read_cnt += 1
                data = self.port.read()
                print(data)
                if len(data) == 0 or read_cnt > max_read:
                    warnings.warn('串口读取超时')
                    self.portClose()
                    break

                if sta == SM.findinghead:
                    if data == self.h:
                        buf += data
                        sta = SM.findingtail
                elif sta == SM.findingtail:
                    buf += data
                    if data == self.t and (len(buf) == self.length or self.length is None):
                        yield buf
                        read_cnt = 0
                        buf = b''
                        sta = SM.findinghead

            # except Exception as e:
            #     print('throw a exception')
            #     print(traceback.format_exc())
            #     # self.portClose()
            #     # print('quited')
            #     # return

    def clear_buf(self):
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()

    @staticmethod
    def list_ports():
        ports = serial.tools.list_ports.comports()
        return list(map(lambda x: (x.device, x.description), ports))

class SM2h2t:
    findinghead1=0
    findinghead2=1
    findingtail=2

class MySerial_2head1tail(MySerial):
    def __init__(self, h2, *args, **kwargs):
        super(MySerial_2head1tail, self).__init__(*args, **kwargs)
        # self.frame_len = frame_len
        self.h2 = h2

    def readData(self):
        buf = b''
        sta = SM2h2t.findinghead1
        read_cnt = 0
        max_read = 64 * 10
        while True:
            read_cnt += 1
            data = self.port.read()
            # print(data.hex())
            if len(data) == 0 or read_cnt > max_read:
                warnings.warn('串口读取超时')
                self.portClose()
                break

            if sta == SM2h2t.findinghead1:
                if data == self.h:
                    # buf += data
                    sta = SM2h2t.findinghead2
            elif sta == SM2h2t.findinghead2:
                if data == self.h2:
                    # buf += data
                    sta = SM2h2t.findingtail
            elif sta == SM2h2t.findingtail:
                if data == self.t and (len(buf) == self.length or self.length is None):
                    yield buf
                    read_cnt = 0
                    buf = b''
                    sta = SM2h2t.findinghead1
                else:
                    buf += data


class MySerial_headerOnly(MySerial):
    def __init__(self, frame_len, *args, **kwargs):
        super(MySerial_headerOnly, self).__init__(*args, **kwargs)
        self.frame_len = frame_len

    def readData(self):
        buf = b''
        sta = SM.findinghead
        cnt = 0
        read_cnt = 0
        max_read = 50

        while True:
            read_cnt += 1
            data = self.port.read()
            # print(data.hex())
            if len(data) == 0 or read_cnt > max_read:
                warnings.warn('串口读取超时')
                self.portClose()
                break

            if sta == SM.findinghead:
                if data == self.h:
                    buf += data
                    sta = SM.findingtail
            elif sta == SM.findingtail:
                buf += data
                cnt += 1
                if cnt == self.frame_len - 1:
                    yield buf
                    buf = b''
                    read_cnt = 0
                    cnt = 0
                    sta = SM.findinghead


    # def start_listen(self):
    #     threading.Thread(target=self.readData).start()

def find_port(dev_class):
    ports = MySerial.list_ports()
    for p, discrb in ports:
        try:
            dev = None
            port_veri = False
            if '蓝牙链接' in discrb or 'bluetooth' in discrb:
                continue
            dev = dev_class(p, timeout=1)
            port_veri = dev.port_verify()
        except:
            # print(traceback.format_exc())
            print(p, 'timeout')
            continue
        finally:
            if dev is not None and dev.serial.port.isOpen():
                dev.clear_port()
                dev.close_port()
            if port_veri:
                return port_veri
    else:
        return None


def find_port_radarlike(dev_class, addrs=None):
    '''
    :param dev_class: a Radar-like device, which is addr needed when asking.
    :param addrs: possible addresses list. default to ['00',]
    :return: (com*, addr)
    '''
    if addrs is None:
        addrs = ['00']
    ports = MySerial.list_ports()
    for p, discrb in ports:
        try:
            dev = None
            port_addrveri = False
            if '蓝牙链接' in discrb or 'bluetooth' in discrb:
                continue
            # 逐个测试addr
            for addr in addrs:
                dev = dev_class(p, addr=addr, timeout=1)
                port_addrveri = dev.is_addr_valid()
        except:
            print(traceback.format_exc())
            print(p, 'timeout')
            continue
        finally:
            if dev is not None and dev.serial.port.isOpen():
                dev.clear_port()
                dev.close_port()
            if port_addrveri:
                return port_addrveri
    else:
        return None



if __name__ == '__main__':
    ats = ['apply_usb_info', 'description', 'device', 'hwid', 'interface', 'location', 'manufacturer', 'name', 'pid', 'product', 'serial_number', 'usb_description', 'usb_info', 'vid']
    ports = list(serial.tools.list_ports.comports())
    print(ports)
    for a in ats:
        p = ports[0]
        print(a, getattr(p, a))
    # for p in ports:
    #     print(p.name)
    # print(MySerial.list_ports())
    # m = MySerial('COM11', timeout=1)
    # m.sendData('hello'.encode())
