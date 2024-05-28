import struct
import warnings
import serial

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

myserial = MySerial_headerOnly(4, 'com4', h=b'\xff', buandRate=9600, timeout=2)

for res in myserial.readData():
    # print(res)
    dis = struct.unpack('>h', res[1:3])[0] / 10.0
    if dis > 500:
        dis = 'MMMMMMMMMMMMMMMMM'
    print(dis)
