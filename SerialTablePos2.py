import time
import serial
import threading

recv_buf_len = 30
frame_start = 0x9B
frame_end = 0x9D
frame_esc = 0x5C

send_nokey = b'\x9B\x06\x02\x00\x00\x6C\xA1\x9D'
send_upkey = b'\x9B\x06\x02\x01\x00\xFC\xA0\x9D'
send_downkey = b'\x9B\x06\x02\x02\x00\x0C\xA0\x9D'
send_prekey1 = b'\x9B\x06\x02\x04\x00\xAC\xA3\x9D'
send_prekey2 = b'\x9B\x06\x02\x08\x00\xAC\xA6\x9D'
send_prekey3 = b'\x9B\x06\x02\x10\x00\xAC\xAC\x9D'
send_showkey = b'\x9B\x07\x12\x3F\x71\x71\xC0\x40\x9D'
led_code = [0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F]

class SerialCommand():
    def __init__(self, portname, baudrate=9600, stopbits=1, bytesize=8):

        # self.portname = portname
        # self.baudrate = baudrate
        # self.stopbits = stopbits
        # self.bytesize = bytesize

        self.running = 1
        self.send_key = send_nokey
        self.height_value = 0

        self.ser = serial.Serial()
        self.ser.port = portname
        self.ser.baudrate = baudrate
        self.ser.stopbits = stopbits
        self.ser.bytesize = bytesize

        # self.send_buf = bytearray(recv_buf_len+1)
        self.recv_buf = bytearray(recv_buf_len + 1)

        self.ser.open()
        self.send_key = send_nokey

        print(self.ser)
        self.last_byte = frame_end
        self.rec_buf_index = 0
        self.recv_buf[self.rec_buf_index] = 0
        self.rec_buf_flag = 0  # = 1 is an valid frame
        self.led_value = 0
        self.ser.reset_input_buffer()
        # print("Opening %s at %u baud" % (self.portname, self.baudrate))
        self.last_key = self.send_key

    def run(self, cmdkey):

        self.send_key = cmdkey
        if self.last_key != self.send_key:
            print("thread:%s" % self.send_key)
            self.last_key = self.send_key

        self.running = 0
        while self.running < 1:
            self.running = self.running + 1
            print(self.running)

            ##                        print("thread: %s" % (self.send_key))

            # 串口处理
            count = self.ser.inWaiting()
            print("count", count)

            for i in range(0, count):
                recv_byte = int.from_bytes(self.ser.read(), 'little')
                if (self.rec_buf_index >= recv_buf_len) or (recv_byte == frame_start and self.last_byte != frame_esc):
                    self.rec_buf_index = 0
                if (recv_byte == frame_start or recv_byte == frame_end or recv_byte == frame_esc) and (
                        self.last_byte == frame_esc):
                    self.recv_buf[self.rec_buf_index] = recv_byte
                else:
                    self.rec_buf_index = self.rec_buf_index + 1
                    self.recv_buf[self.rec_buf_index] = recv_byte
                    if recv_byte == frame_end:
                        self.rec_buf_flag = 1
                self.last_byte = recv_byte

                if self.rec_buf_flag == 1:  # 接收到一个完整数据帧
                    ##                                        print(rec_buf_index)
                    if self.rec_buf_index > 5:
                        ##                                                for j in range(0,rec_buf_index):
                        ##                                                        print(hex(self.recv_buf[j]),end=" ")
                        ##                                                print("\n")
                        if self.recv_buf[3] == 0x11:  # 如果是询问键值
                            sc = self.ser.write(self.send_key)
                        ##                                                        print("sent",sc,"bytes")
                        elif self.recv_buf[3] == 0x12:  # 如果是显示指令
                            led100 = self.recv_buf[4] & 0x7F
                            led10 = self.recv_buf[5] & 0x7F
                            led1 = self.recv_buf[6] & 0x7F
                            if (led100 in led_code) and (led10 in led_code) and (led1 in led_code):
                                self.led_value = led_code.index(led100) * 100
                                self.led_value = self.led_value + led_code.index(led10) * 10
                                self.led_value = self.led_value + led_code.index(led1)
                                if self.recv_buf[4] & 0x80:
                                    self.led_value = self.led_value / 100
                                elif self.recv_buf[5] & 0x80:
                                    self.led_value = self.led_value / 10
                                print("Height=", self.led_value)
                                self.height_value = self.led_value
                            else:
                                print("Disp Err")

                    rec_buf_flag = 0
                    rec_buf_index = 0
# send_nokey = b'\x9B\x06\x02\x00\x00\x6C\xA1\x9D'
# send_upkey = b'\x9B\x06\x02\x01\x00\xFC\xA0\x9D'
# send_downkey = b'\x9B\x06\x02\x02\x00\x0C\xA0\x9D'
# send_prekey1 = b'\x9B\x06\x02\x04\x00\xAC\xA3\x9D'    # 1挡位
# send_prekey2 = b'\x9B\x06\x02\x08\x00\xAC\xA6\x9D'    # 2挡位
# send_prekey3 = b'\x9B\x06\x02\x10\x00\xAC\xAC\x9D'    # 3挡位


class TableControl(threading.Thread):
    def __init__(self,name = None):
        threading.Thread.__init__(self,name = None)
        self.current_height = 0
        self.isinit = 0
        self.targetStatus = -1
        self.trig_stop = False


    def run(self):
            while not self.trig_stop:
                self.loopRun()
                time.sleep(0.1)
            self.trig_stop = False

    def loopRun(self):
        if self.targetStatus == 0:  # 低位
            print("低位")
            self.controller.run(send_prekey1)
            if self.controller.height_value != self.current_height:
                self.current_height = self.controller.height_value
            self.targetStatus = -1
        elif self.targetStatus == 1:  # 中位
            print("中位")
            self.controller.run(send_prekey2)
            if self.controller.height_value != self.current_height:
                self.current_height = self.controller.height_value
            self.targetStatus = -1
        elif self.targetStatus == 2:  # 高位
            print("高位")
            self.controller.run(send_prekey3)
            if self.controller.height_value != self.current_height:
                current_height = self.controller.height_value
            self.targetStatus = -1
        else:
            self.controller.run(send_nokey)

    def initCom(self, port):
        self.controller = SerialCommand(port)

    # def getHeight(self):
    #     while self.controller.height_value == 0:
    #         self.controllerserth.run(send_downkey)
    #     return self.controller.height_value
    #
    # def setHeight(self, target_height):
    #     currentHeight = self.getHeight()
    #     if currentHeight < target_height:  # 上升
    #         while self.controller.height_value < target_height:
    #             self.controller.run(send_upkey)
    #     if currentHeight > target_height:  # 下降
    #         while self.controller.height_value > target_height:
    #             self.controller.run(send_downkey)

    def setHeightStatus(self):
        while True:
            time.sleep(0.1)
            if self.targetStatus == 0:  # 低位
                time.sleep(0.1)
                print("低位")
                self.controller.run(send_prekey1)
                print(self.current_height, self.controller.height_value)
                if self.controller.height_value != self.current_height:
                    self.current_height = self.controller.height_value
            # self.targetStatus = -1
            elif self.targetStatus == 1:  # 中位
                time.sleep(0.1)
                print("中位")
                self.controller.run(send_prekey2)
                if self.controller.height_value != self.current_height:
                    self.current_height = self.controller.height_value
            elif self.targetStatus == 2:  # 高位
                time.sleep(0.1)
                print("高位")
                self.controller.run(send_prekey3)
                if self.controller.height_value != self.current_height:
                    current_height = self.controller.height_value
            else:
                pass

    def closeCom(self):
        self.controller.ser.close()

# controller = SerialCommand('COM7')
# controller.run(send_prekey1)
# time.sleep(10)
# controller.run(send_nokey)