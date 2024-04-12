# -*- coding: utf-8 -*-
from drivers.Serial import *
import cv2
import queue
from comps.utils import *
from comps.IRdataCollect import *
from comps.SonicDataCollect import *
from my_socket import server
from argparse import ArgumentParser
import math
import sys
from PyQt6 import QtWidgets
from Ui_serialShow import Ui_SerialShow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMessageBox,QFileDialog,QInputDialog,QWidget
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
import serial.tools.list_ports
from models.infer import MyInference
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QDialog
from PyQt6.QtGui import QIcon
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
myserial = None



class DEVICE:
    IR_data_collector = IRDataCollect()
    sonic_device1 = SonicDataCollect(MESSAGE.sonic1, SOCKET.sonic1, 'sonic1')
    predictor = MyInference()


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


       

# 主界面类
class MyMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_SerialShow()
        self.ui.setupUi(self)
        # 检测串口
        self.ui.pushButton_cleckcoms.clicked.connect(self.click_pushButton_cleckcoms)
        # 快速打开保存文件夹
        self.ui.pushButton_quickopen.clicked.connect(self.click_pushButton_quickopen)
        # 显示数据
        self.ui.pushButton_show.clicked.connect(self.click_pushButton_show)
        # 开始记录
        self.ui.pushButton_start.clicked.connect(self.clickpushButton_start)
        # 新增人员
        self.ui.pushButton_addperson.clicked.connect(self.clickpushButton_add)
        
        # SonicDataCollect初始化与信号槽链接
        self.sonic_data_collector = SonicDataCollect(MESSAGE.sonic1, SOCKET.sonic1, 'sonic1')
        self.sonic_data_collector.new_dist_signal.connect(self.update_distance_display)
        # IRDataCollect初始化与信号槽链接
        self.ir_data_collector = IRDataCollect()        
        self.ir_data_collector.new_heatmap_signal.connect(self.update_heatmap_display)
        # MyInference初始化与信号槽链接
        self.predictor_result = MyInference()
        self.predictor_result.predict_result_signal.connect(self.update_predict_result)

        # 选择模式
        self.ui.comboBox_mode.currentTextChanged.connect(self.on_mode_changed)
        # 选择人员
        self.ui.comboBox_chooseperson.currentTextChanged.connect(self.on_chooseperson_changed)
        # 选择高低位
        self.ui.comboBox_chooseposition.currentTextChanged.connect(self.on_chooseposition_changed)
        # 选择姿态
        self.ui.comboBox_chooseposture.currentTextChanged.connect(self.on_chooseposture_changed)
        # 选择推理时桌子位置
        self.ui.comboBox_predictPos.currentTextChanged.connect(self.on_tablepos_changed)
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # 是否连接Com
        self.iscomdata = 0
        # 记录时间
        self.secondSec = 0

        # 初始化各ComboBox中的值
        SCENETYPEPos=['低位','高位']
        SCENETYPEModes = \
            ['坐姿'
            ,'其他坐姿'
            , '站姿'
            ,'其他站姿'
            , '坐姿到站姿'
            , '站姿到坐姿'
            , '无人']

        nametypes = self.getnames('./persondata')
        for pos_type in SCENETYPEPos:
                    self.ui.comboBox_chooseposition.addItem(pos_type)
        for mode_type in SCENETYPEModes:
                    self.ui.comboBox_chooseposture.addItem(mode_type)
        for name_type in nametypes:
                    self.ui.comboBox_chooseperson.addItem(name_type)
        self.allname = []

        # 刷新人员信息显示
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.timer_callback)
        self.refresh_timer.start(500)

        # 刷新记录时间计时
        self.caltimer = QtCore.QTimer(self)
        self.caltimer.setInterval(100)
        self.caltimer.timeout.connect(self.onTimerOut)

       
        self.issaving = 0                                        # 是否正在记录，默认为否
        self.ui.stackedWidget_Mode.setCurrentIndex(0)            # 默认为采集模式   
        self.ui.comboBox_predictPos.setCurrentIndex(0)           # 默认桌子为低位
        self.predictor_result.set_table_position(0)              # 默认桌子为低位
        self.wordThreads = []  
        self.predict_pos = []

    # 刷新记录时间
    def onTimerOut(self):
        self.secondSec+=1
        self.ui.pushButton_start.setText("停止记录:"+str(self.secondSec/10.0)+"s")


        
        

    def keyPressEvent(self, keyevent):        
        current_mode = self.ui.comboBox_chooseposture.currentIndex()
        if keyevent.key() == Qt.Key.Key_Space or keyevent.key() == Qt.Key.Key_Return:
            self.clickpushButton_start()        
        elif  keyevent.key() == Qt.Key.Key_Right and current_mode != self.ui.comboBox_chooseposture.count() - 1:            
            self.ui.comboBox_chooseposture.setCurrentIndex(current_mode + 1)
        elif  keyevent.key() == Qt.Key.Key_Left and current_mode != 0:            
            self.ui.comboBox_chooseposture.setCurrentIndex(current_mode - 1)
        elif  keyevent.key() == Qt.Key.Key_Up or keyevent.key() == Qt.Key.Key_Down:      
            self.ui.comboBox_chooseposition.setCurrentIndex(not self.ui.comboBox_chooseposition.currentIndex())
        else:
            pass

    # 选择推理时桌子位置
    def on_tablepos_changed(self, text):
        if(text == "高位"):
            self.predictor_result.set_table_position(1)
        elif(text == "低位"):
            self.predictor_result.set_table_position(0)
    
    # 选择模式
    def on_mode_changed(self, text):
        self.ui.stackedWidget_Mode.setCurrentIndex(self.ui.comboBox_mode.currentIndex())        
        if self.ui.stackedWidget_Mode.currentIndex() == 0:
            CONTROL.TESTING = False
            self.predictor_result.threadon = 0
            self.ui.label_result.setText('')
            self.ui.label_action.setText('')
            
                
        if self.ui.stackedWidget_Mode.currentIndex() == 1:
            CONTROL.TESTING = True
            self.predictor_result.threadon = 1
            self.ui.label_result.setText('')
            self.ui.label_action.setText('')     

                      
            

    # 选择姿态
    def on_chooseposture_changed(self,text):
        self.setFocus()
    # 选择人员
    def on_chooseperson_changed(self,text):
        self.setFocus()
    # 选择高低位
    def on_chooseposition_changed(self,text):
        self.setFocus()
    

    def arrays_equal(self,arr1,arr2):
        if len(arr1) != len(arr2):
            return 0
        for i in range(len(arr1)):
            if arr1[i] not in arr2:
                return 0
        for i in range(len(arr2)):
            if arr2[i] not in arr1:
                return 0
        return 1
    
    # 刷新人员信息显示
    def timer_callback(self):
        names = self.getnames('./persondata')
        current_person = self.ui.comboBox_chooseperson.currentText()
        if not self.arrays_equal(names,self.allname):
            print("存在姓名更改")
            self.ui.comboBox_chooseperson.clear()
            for per_name in names:
                self.ui.comboBox_chooseperson.addItem(per_name)
            if current_person in names:
                self.ui.comboBox_chooseperson.setCurrentText(current_person)
        self.allname = names
        
    # 检测串口   
    def click_pushButton_cleckcoms(self):
        self.ui.comboBox_Coms.clear()
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            self.ui.comboBox_Coms.addItem(f"{port.device}")

    # 快速打开保存文件夹
    def click_pushButton_quickopen(self):
        start_directory = pl.Path('./persondata')
        os.system("explorer.exe %s" % start_directory)

    # 显示数据
    def click_pushButton_show(self):
        if self.ui.comboBox_Coms.count()==0:
            QMessageBox.information(self, "提示","未找到端口连接！")
            return
        try:
            global myserial
            parser = ArgumentParser()
            parser.add_argument('--port', type=int, default=SOCKET.SERVER_PORT)
            parser.add_argument('--serial', type=str, default=self.ui.comboBox_Coms.currentText())
            args = parser.parse_args()
            SOCKET.SERVER_PORT = args.port
            myserial = MySerial_2head1tail(b'\xFA', args.serial, b'\xAF', b'\xFF', length=[64 * 4 + 1, 5])            
            self.ui.pushButton_cleckcoms.setEnabled(0)
            self.ui.comboBox_Coms.setEnabled(0)
        except Exception as e:
            QMessageBox.information(self, "提示","串口连接失败！")
            return
        
        self.ui.pushButton_show.setEnabled(False)
        jobs = [message_classify, self.sonic_data_collector.play_sonic, self.ir_data_collector.play_IR]
        for job in jobs:
            t = threading.Thread(target=job,)
            t.daemon = True
            t.start()

        t_predice = threading.Thread(target=self.predictor_result.get_action,)
        t_predice.daemon = True
        t_predice.start()
        
        self.iscomdata = 1

        
    # 新增人员
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

    

    def get_savepath_personal(self,filename):
        button = QMessageBox.question(self, "提示", "是否保存")
        if button == QMessageBox.StandardButton.Yes:
            person_name = self.ui.comboBox_chooseperson.currentText()
            table_pos = self.ui.comboBox_chooseposition.currentText()
            person_pos = self.ui.comboBox_chooseposture.currentText()
            sceneroot = pl.Path('./persondata') / pl.Path('./' + person_name)/pl.Path('./' + table_pos) / pl.Path('./' + person_pos)
            
            os.makedirs(sceneroot, exist_ok=True)
            return [1,filename, sceneroot]
        else:
            return False

    def get_savepath_training(self,filename):
        table_pos = self.ui.comboBox_chooseposition.currentText()
        person_pos = self.ui.comboBox_chooseposture.currentText()
        save_path2 = ""
        if table_pos == "高位" and person_pos == "无人":
            save_path2 = "high-position-nobody"
        if table_pos == "高位" and person_pos == "站姿":
            save_path2 = "high-position-stand"
        if table_pos == "高位" and person_pos == "站姿到坐姿":
            save_path2 = "high-positon-stand2sit"
        if table_pos == "高位" and person_pos == "坐姿":
            save_path2 = "high-position-sit"
        if table_pos == "高位" and person_pos == "坐姿到站姿":
            save_path2 = "high-position-sit2stand"
        if table_pos == "低位" and person_pos == "无人":
            save_path2 = "low-position-nobody"
        if table_pos == "低位" and person_pos == "站姿":
            save_path2 = "low-position-stand"
        if table_pos == "低位" and person_pos == "站姿到坐姿":
            save_path2 = "low-positon-stand2sit"
        if table_pos == "低位" and person_pos == "坐姿":
            save_path2 = "low-position-sit"
        if table_pos == "低位" and person_pos == "坐姿到站姿":
            save_path2 = "low-position-sit2stand"
        if table_pos == "低位" and person_pos == "其他坐姿":
            save_path2 = "low-position-otheresit"
        if table_pos == "高位" and person_pos == "其他坐姿":
            save_path2 = "high-position-othersit"
        if table_pos == "低位" and person_pos == "其他站姿":
            save_path2 = "low-position-otherestand"
        if table_pos == "高位" and person_pos == "其他站姿":
            save_path2 = "high-position-otherestand"

        trainroot = pl.Path('./data')/ pl.Path('./' + save_path2)
        return [1,filename, trainroot]

    # 开始记录
    def clickpushButton_start(self):
        if self.issaving == 0:
            if self.ui.comboBox_chooseperson.currentText() and self.ui.comboBox_chooseposition.currentText() and self.ui.comboBox_chooseposture.currentText():
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
            self.ui.comboBox_chooseposture.setEnabled(0)
            CONTROL.RECORDING = 1
            print(f'recording now {CONTROL.RECORDING}')
            self.issaving = 1
            self.secondSec = 0
            self.caltimer.start()
        else:
            self.caltimer.stop()
            CONTROL.RECORDING = 0
            if not CONTROL.RECORDING and len(self.ir_data_collector.IR_imgs) > 0:
                timestamp = int(time.time())
                person_name = self.ui.comboBox_chooseperson.currentText()
                table_pos = self.ui.comboBox_chooseposition.currentText()
                person_pos = self.ui.comboBox_chooseposture.currentText()
                file_name = "new" + person_name + "_" + table_pos + "_" + person_pos + "_" +f'{timestamp}'

                save_args1 = self.get_savepath_personal(file_name)
                save_args2 = self.get_savepath_training(file_name)
                if save_args1 is False:
                    # discard data
                    self.ir_data_collector.clear_buffer()
                    self.sonic_data_collector.clear_buffer()
                else:
                    self.ir_data_collector.save_data(*save_args1)
                    self.sonic_data_collector.save_data(*save_args1)
                    self.ir_data_collector.save_data(*save_args2)
                    self.sonic_data_collector.save_data(*save_args2)
                    self.ir_data_collector.clear_buffer()
                    self.sonic_data_collector.clear_buffer()          

                    # CONTROL.update_lastround(save_args)
            self.ui.comboBox_chooseperson.setEnabled(1)
            self.ui.comboBox_chooseposition.setEnabled(1)
            self.ui.comboBox_chooseposture.setEnabled(1)
            self.issaving = 0
            self.ui.pushButton_start.setText("开始记录")
            

    # SonicDataCollect数据传输
    def update_distance_display(self, distance):
        if distance >= 37999.000:
            self.ui.label_Sonic.setText("超声数据：" + " " * 28 + '-----')
        else:
            self.ui.label_Sonic.setText("超声数据："+" "*24 + '%9.3f' % distance)

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

    def update_predict_result(self,result):
        # 'idle', 'sit', 'sit2stand', 'stand', 'stand2sit'
        if  result[0] == 'idle':
            self.ui.label_result.setText('无人')
        elif result[0] == 'sit':
            self.ui.label_result.setText('坐姿')
        elif result[0] == 'stand':
            self.ui.label_result.setText('站姿')
        elif result[0] == 'sit2stand':
            self.ui.label_result.setText('坐姿---站姿')
        elif result[0] == 'stand2sit':
            self.ui.label_result.setText('站姿---坐姿')

        self.ui.label_action.setText(result[1])
        self.predict_pos = result[2][0]

        plt.ion() 
        plt.clf()
        plt.figure(1)
        plt.ylim(0,1)
        x = ['idle','sit','sit2stand','stand','stand2sit']        
        plt.plot(x,self.predict_pos,'g-.o')
        plt.show()
        
        
          


    # IRDataCollect数据传输
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
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec())
