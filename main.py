from drivers.Serial import *
import cv2
import queue
from comps.utils import *
from comps.IRdataCollect import *
from comps.SonicDataCollect import *
from my_socket import server
from argparse import ArgumentParser
import sys
from PyQt6 import QtWidgets
from serialShow import Ui_SerialShow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMessageBox,QFileDialog,QInputDialog
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
import serial.tools.list_ports
import math
myserial = None


class DEVICE:
    IR_data_collector = IRDataCollect()
    sonic_device1 = SonicDataCollect(MESSAGE.sonic1, SOCKET.sonic1, 'sonic1')


def message_classify():
    for res in myserial.readData():
        # print(res)
        paras = (res[1:], True, None)
        try:
            if res[0] == TAG.IR and not MESSAGE.IR.full():
                MESSAGE.IR.put(*paras)
            elif res[0] == TAG.SONIC1 and not MESSAGE.sonic1.full():
                MESSAGE.sonic1.put(*paras)
        except Exception as e:
            traceback.print_exc()
            break


'''
The key are mostly detected in IRdataCollect.playIR(), because the key is obtained by cv2
'''
def key_handler():
    while True:
        key = MESSAGE.KEY.get()
        if key == 32:
            # space
            CONTROL.RECORDING = not CONTROL.RECORDING
            print(f'recording now {CONTROL.RECORDING}')

            if not CONTROL.RECORDING and len(DEVICE.IR_data_collector.IR_imgs) > 0:
                args = CONTROL.get_scenetype()
                if args is False:
                    # discard data
                    DEVICE.IR_data_collector.clear_buffer()
                    DEVICE.sonic_device1.clear_buffer()
                else:
                    DEVICE.IR_data_collector.save_data(*args)
                    DEVICE.sonic_device1.save_data(*args)
                    # save some parameters to instance
                    CONTROL.update_lastround(args)

        elif key == ord('q'):
            break

        # elif key == 27:
        #     # esc, to re-save the last round file to another directory
        #     args = CONTROL.get_scenetype()
        #     if args is False:
        #         # discard data
        #         DEVICE.IR_data_collector.clear_buffer()
        #         DEVICE.sonic_device1.clear_buffer()
        #     else:
        #         new_filename = CONTROL.last_filename.replace(CONTROL.last_scenetype, args[0])
        #         args[1] = new_filename
        #
        #         DEVICE.IR_data_collector.resave_data(*args)
        #         DEVICE.sonic_device1.resave_data(*args)
        #         # save some parameters to instance
        #         CONTROL.update_lastround(args)
        #         # another change for re-saving the files
        #         print("上一轮的存储是否想改变主意？按下ESC以重新保存，否则请忽略。")



def main():
    jobs = [server.start_server, message_classify, DEVICE.sonic_device1.play_sonic, DEVICE.IR_data_collector.play_IR]
    my_threads = []
    for job in jobs:
        t = threading.Thread(target=job,)
        my_threads.append(t)
        t.daemon = True
        t.start()
    print(f'System started with {len(jobs)} threads.')
    key_handler()

class MyMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_SerialShow()
        self.ui.setupUi(self)
        self.ui.pushButton_opencom.clicked.connect(self.click_pushButton_opencom)
        self.ui.pushButton_quickopen.clicked.connect(self.click_pushButton_quickopen)
        self.ui.pushButton_show.clicked.connect(self.click_pushButton_show)
        self.ui.pushButton_start.clicked.connect(self.clickpushButton_start)
        self.ui.pushButton_addperson.clicked.connect(self.clickpushButton_add)
        self.sonic_data_collector = SonicDataCollect(MESSAGE.sonic1, SOCKET.sonic1, 'sonic1')
        self.sonic_data_collector.new_dist_signal.connect(self.update_distance_display)
        self.ui.comboBox_mode.currentTextChanged.connect(self.on_mode_changed)
        self.ui.spinBox_IInterval.valueChanged.connect(self.on_spinBox_IRChanged)
        self.ui.spinBox_SInterval.valueChanged.connect(self.on_spinBox_SonicChanged)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ui.pushButton.clicked.connect(self.click_pushButton1)
        self.iscomdata = 0
        self.ir_data_collector = IRDataCollect()
        self.ir_data_collector.new_heatmap_signal.connect(self.update_heatmap_display)
        SCENETYPEPos=['低位','高位']
        SCENETYPEModes = \
            ['坐姿'
            , '站姿'
            , '坐姿到站姿'
            , '站姿到坐姿'
            , '无人']

        nametypes = self.getnames('./persondata')
        for pos_type in SCENETYPEPos:
                    self.ui.comboBox_chooseposition.addItem(pos_type)
        for mode_type in SCENETYPEModes:
                    self.ui.comboBox_choosemode.addItem(mode_type)
        for name_type in nametypes:
                    self.ui.comboBox_chooseperson.addItem(name_type)
        self.allname = []
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.timer_callback)
        self.refresh_timer.start(500)
        self.issaving = 0
        self.ui.stackedWidget_Mode.setCurrentIndex(0)
        self.IRDatas = []
        self.IRIMGDatas = []
        self.SONICDatas = []
        self.picknumber = 0
        self.isfull = 0
    def keyPressEvent(self, keyevent):
        if keyevent.key() == Qt.Key.Key_S:
            print("click")
            self.clickpushButton_start()

    def on_spinBox_IRChanged(self):
        print('红外间隔为:', self.ui.spinBox_IInterval.text())
        CONTROL.IR_interval = int(self.ui.spinBox_IInterval.text())
        self.isfull = 0
        self.IRDatas.clear()
        self.IRIMGDatas.clear()
        self.SONICDatas.clear()
    def on_spinBox_SonicChanged(self):
        print('超声间隔为:', self.ui.spinBox_SInterval.text())
        CONTROL.Sonic_interval = int(self.ui.spinBox_SInterval.text())
        self.isfull = 0
        self.IRDatas.clear()
        self.IRIMGDatas.clear()
        self.SONICDatas.clear()

    def on_mode_changed(self, text):
        self.ui.stackedWidget_Mode.setCurrentIndex(self.ui.comboBox_mode.currentIndex())
        self.isfull = 0
        self.IRDatas.clear()
        self.IRIMGDatas.clear()
        self.SONICDatas.clear()
        if self.ui.stackedWidget_Mode.currentIndex() == 0:
            print("# 采集模式")
            CONTROL.TESTING = False
        if self.ui.stackedWidget_Mode.currentIndex() == 1:
            print("# 推理模式")
            CONTROL.TESTING = True

    def timer_callback(self):
        names = self.getnames('./persondata')
        if names != self.allname:
            self.ui.comboBox_chooseperson.clear()
            for per_name in names:
                self.ui.comboBox_chooseperson.addItem(per_name)
        self.allname = names

        if self.ui.stackedWidget_Mode.currentIndex() == 1:  # 推理模式
            self.showTestResult()



    def showTestResult(self):
        if len(self.ir_data_collector.IR_imgs) > 0 and len(self.sonic_data_collector.distances) > 0:
            IRdata = self.ir_data_collector.IR_imgs
            IRImgdata = self.ir_data_collector.heat_imgs
            Sonicdata = self.sonic_data_collector.distances

            if self.isfull == 1:
                self.IRDatas = self.IRDatas[(len(self.IRDatas)//6):]
                self.IRIMGDatas = self.IRIMGDatas[(len(self.IRIMGDatas)//6):]
                self.SONICDatas = self.SONICDatas[(len(self.SONICDatas) // 6):]

            self.IRDatas.extend(IRdata)
            self.SONICDatas.extend(Sonicdata)
            self.IRIMGDatas.extend(IRImgdata)

            self.ir_data_collector.clear_buffer()
            self.sonic_data_collector.clear_buffer()

            print("len(self.IRDatas)=",len(self.IRDatas))
            print("len(self.IRIMGDatas)=",len(self.IRIMGDatas))
            print("len(self.SONICDatas)=",len(self.SONICDatas))
            print("升降桌位置:",self.ui.comboBox_2.currentText())


            if self.picknumber == 5:
                self.picknumber = 0
                self.isfull=1
            else:
                self.picknumber += 1

    def click_pushButton1(self):
        #保存推理原数据
        save_args = self.get_savetype()
        if save_args is False:
            # discard data
            self.ir_data_collector.clear_buffer()
            self.sonic_data_collector.clear_buffer()
        else:
            print("Saving IR data...")

            # save IR_img np.array as mat
            sio.savemat(f'{save_args[2] / save_args[1]}.mat',
                        {'IR_video': np.array(self.IRDatas)}, appendmat=True)

            # save preview heatmap videos
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
            video_writer = cv2.VideoWriter(f'{save_args[2] / save_args[1]}.mp4', fourcc, 10.0, (160, 160))
            for hmap in self.IRIMGDatas:
                video_writer.write(hmap)
            video_writer.release()

            CONTROL.update_lastround(save_args)

    def click_pushButton_opencom(self):
        ports = list(serial.tools.list_ports.comports())
        try:
            global myserial
            parser = ArgumentParser()
            parser.add_argument('--port', type=int, default=SOCKET.SERVER_PORT)
            parser.add_argument('--serial', type=str, default='COM3')
            args = parser.parse_args()
            SOCKET.SERVER_PORT = args.port
            myserial = MySerial_2head1tail(b'\xFA', args.serial, b'\xAF', b'\xFF', length=[64 * 4 + 1, 5])
            self.ui.pushButton_opencom.setEnabled(0)
        except Exception as e:
            QMessageBox.information(self, "提示","串口连接失败")

    def click_pushButton_quickopen(self):
        start_directory = pl.Path('./persondata')
        os.system("explorer.exe %s" % start_directory)

    def click_pushButton_show(self):
        self.ui.pushButton_show.setEnabled(False)
        t = threading.Thread(target=message_classify)
        t.daemon = True
        t.start()

        t1 = threading.Thread(target = self.sonic_data_collector.play_sonic)
        t1.daemon =True
        t1.start()

        t2 = threading.Thread(target = self.ir_data_collector.play_IR)
        t2.daemon =True
        t2.start()
        self.iscomdata = 1

    def clickpushButton_add(self):
        name, ok = QInputDialog.getText(self, '新增测试人员', '请输入您的名字:')
        if ok:
            names = self.getnames('./persondata')
            if name in names:
                QMessageBox.information(self, "提示", "已有该人员，请勿重复输入!")
                return
            nameroot = pl.Path('./persondata') / pl.Path('./' + name)
            os.makedirs(nameroot, exist_ok=True)
            self.ui.comboBox_chooseperson.clear()

            for name_type in self.getnames('./persondata'):
                self.ui.comboBox_chooseperson.addItem(name_type)
            self.ui.comboBox_chooseperson.setCurrentText(name)

    def getnames(self,directory):
        folders = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                folders.append(item)
        return folders

    def get_savetype(self):
            button = QMessageBox.question(self, "提示", "是否保存")
            if button == QMessageBox.StandardButton.Yes:
                person_name = self.ui.comboBox_chooseperson.currentText()
                table_pos = self.ui.comboBox_chooseposition.currentText()
                person_pos = self.ui.comboBox_choosemode.currentText()
                sceneroot = pl.Path('./persondata') / pl.Path('./' + person_name)/pl.Path('./' + table_pos) / pl.Path('./' + person_pos)
                timestamp = int(time.time())
                filename = f'{timestamp}'
                os.makedirs(sceneroot, exist_ok=True)
                return [1,filename, sceneroot]
            else:
                return False


    def clickpushButton_start(self):
        if self.issaving == 0:
            if self.ui.comboBox_chooseperson.currentText() and self.ui.comboBox_chooseposition.currentText() and self.ui.comboBox_choosemode.currentText():
                pass
            else:
                QMessageBox.information(self, "提示", "请输入保存信息!")
                return
            if self.iscomdata == 1:
                pass
            else:
                QMessageBox.information(self, "提示", "请打开串口!")
                return
            self.ui.comboBox_chooseperson.setEnabled(0)
            self.ui.comboBox_chooseposition.setEnabled(0)
            self.ui.comboBox_choosemode.setEnabled(0)
            CONTROL.RECORDING = 1
            print(f'recording now {CONTROL.RECORDING}')
            self.issaving = 1
            self.ui.pushButton_start.setText("停止记录")
        else:
            CONTROL.RECORDING = 0
            if not CONTROL.RECORDING and len(self.ir_data_collector.IR_imgs) > 0:
                save_args = self.get_savetype()
                if save_args is False:
                    # discard data
                    self.ir_data_collector.clear_buffer()
                    self.sonic_data_collector.clear_buffer()
                else:
                    self.ir_data_collector.save_data(*save_args)
                    self.sonic_data_collector.save_data(*save_args)
                    CONTROL.update_lastround(save_args)
            self.ui.comboBox_chooseperson.setEnabled(1)
            self.ui.comboBox_chooseposition.setEnabled(1)
            self.ui.comboBox_choosemode.setEnabled(1)
            self.issaving = 0
            self.ui.pushButton_start.setText("开始记录")





    def update_distance_display(self, distance):
        if distance >= 37999.000:
            self.ui.label_Sonic.setText("超声数据：" + " " * 54 + '-----')
        else:
            self.ui.label_Sonic.setText("超声数据："+" "*50 + '%9.3f' % distance)

    def cvMatToQImage(self, cvMat):
        if len(cvMat.shape) == 2:
            # 灰度图是单通道，所以需要用Format_Indexed8
            rows, columns = cvMat.shape
            bytesPerLine = columns
            return QImage(cvMat.data, columns, rows, bytesPerLine, QImage.Format.Format_Indexed8)
        else:
            rows, columns, channels = cvMat.shape
            bytesPerLine = channels * columns
            return QImage(cvMat.data, columns, rows, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()

    def update_heatmap_display(self,heatmap):
        # cv2.imshow('IR_img', heatmap)
        label_width = self.ui.IR_showLabel.width()
        label_height = self.ui.IR_showLabel.height()
        # 调整图像大小为label的大小
        resized_image = cv2.resize(heatmap, (label_width, label_height))

        qimg = self.cvMatToQImage(resized_image)
        pixmap = QPixmap.fromImage(qimg)
        self.ui.IR_showLabel.setPixmap(pixmap)
        self.ui.IR_showLabel.show()



if __name__ == "__main__":

    # try:
    #     main()
    # except Exception as e:
        # traceback.print_exc()
        # myserial.portClose()

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec())

