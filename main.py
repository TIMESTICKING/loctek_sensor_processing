# -*- coding: utf-8 -*-
from drivers.Serial import *
import cv2
from comps.utils import *
from comps.IRdataCollect import *
from comps.SonicDataCollect import *
from argparse import ArgumentParser
import sys
from SerialTablePos import TableControl
from PyQt5 import QtWidgets
from ConnectDeviceDialog import ConnectDeviceDialog
from serialShow import Ui_SerialShow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QInputDialog
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from draw_plot import DrawPlotWidget
import serial.tools.list_ports
from models.infer import InferenceFormula, InferenceTorch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from matplotlib import pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'D:\\Environments\\Anaconda3\\envs\\pytorch_env\\Lib\\site-packages\\PyQt5\\Qt5\\plugins'


class DEVICE:
    IR_data_collector = IRDataCollect()
    sonic_device1 = SonicDataCollect(MESSAGE.sonic1, SOCKET.sonic1, 'sonic1')
    predictor = InferenceTorch() if USE_TORCH else InferenceFormula() # MyInference()


# def plot_show():
#     try:
#         if len(CONTROL.label_percent) != 0:
#             plt.ion()
#             plt.clf()
#             plt.figure(1)
#             plt.ylim(0,1)
#             x = ['idle','sit','sit2stand','stand','stand2sit']
#             plt.plot(x,CONTROL.label_percent,'g-.o')
#             plt.show()
#         else:
#             pass
#     except Exception as e:
#         traceback.print_exc()
#
class MySerial_2head1tail(MySerial):
    def __init__(self, h2, *args, **kwargs):
        super(MySerial_2head1tail, self).__init__(*args, **kwargs)
        # self.frame_len = frame_len
        self.h2 = h2
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
        self.writeData("{'cmd':'SAC','debug': false ,'spit': true }")


    def message_classify(self):
        for res in self.readData():
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
        self.ui.pushButton_checkcoms.clicked.connect(self.click_pushButton_checkcoms)
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
        self.predictor_result = InferenceTorch() if USE_TORCH else InferenceFormula()  # MyInference()
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
        self.__last_changed_height = 0
        # 初始化各ComboBox中的值
        SCENETYPEPos = ['低位', '高位']
        SCENETYPEModes = \
            ['坐姿'
                , '其他坐姿'
                , '站姿'
                , '其他站姿'
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

        # 刷新显示人员名称
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.timer_callback)
        self.refresh_timer.start(500)

        # 控制升降桌升起、落下
        self.tableController = TableControl()
        self.tableDataActivate = False

        # 刷新显示升降桌高度
        self.refresh_tableheight_timer = QtCore.QTimer(self)
        self.refresh_tableheight_timer.timeout.connect(self.refresh_tableheight_timer_callback)
        self.refresh_tableheight_timer.start(100)

        # 刷新记录时间计时
        self.caltimer = QtCore.QTimer(self)
        self.caltimer.setInterval(100)
        self.caltimer.timeout.connect(self.onTimerOut)

        self.issaving = 0  # 是否正在记录，默认为否
        self.ui.stackedWidget_Mode.setCurrentIndex(0)  # 默认为采集模式
        self.ui.comboBox_predictPos.addItem("未知")
        self.predictor_result.set_table_position(0)  # 默认桌子为低位
        self.workThreads = []
        self.predict_pos = []
        self.setFixedSize(1500, 750)
        self.DrawPlot = DrawPlotWidget()
        self.com_table = ""
        self.com_device = ""
        self.lastResult = ""
        self.predictequalTimes = 0
        self.lastSendHeight = -1
        self.com_info = {}

    # 刷新记录时间
    def onTimerOut(self):
        self.secondSec += 1
        self.ui.pushButton_start.setText("停止记录:" + str(self.secondSec / 10.0) + "s")

    def closeEvent(self, *args, **kwargs):  # real signature unknown
        if self.tableController is not None:
            if self.tableController.isinit == True and self.tableController.trig_stop == False:
                print("升降桌断开")
                self.tableController.stop()

        self.myserial.stop_thread()
        self.sonic_data_collector.stop_thread()
        self.ir_data_collector.stop_thread()
        self.predictor_result.stop_thread()
        self.DrawPlot.close()

    def keyPressEvent(self, keyevent):
        current_mode = self.ui.comboBox_chooseposture.currentIndex()
        if keyevent.key() == Qt.Key.Key_Space or keyevent.key() == Qt.Key.Key_Return:
            self.clickpushButton_start()
        elif keyevent.key() == Qt.Key.Key_Right and current_mode != self.ui.comboBox_chooseposture.count() - 1:
            self.ui.comboBox_chooseposture.setCurrentIndex(current_mode + 1)
        elif keyevent.key() == Qt.Key.Key_Left and current_mode != 0:
            self.ui.comboBox_chooseposture.setCurrentIndex(current_mode - 1)
        elif keyevent.key() == Qt.Key.Key_Up or keyevent.key() == Qt.Key.Key_Down:
            self.ui.comboBox_chooseposition.setCurrentIndex(not self.ui.comboBox_chooseposition.currentIndex())
        else:
            pass

    # 选择推理时桌子位置
    def on_tablepos_changed(self, text):
        if (text == "高位"):
            print("模式切换为高位")
            self.predictor_result.set_table_position(1)
        elif (text == "低位"):
            print("模式切换为低位")
            self.predictor_result.set_table_position(0)

    # 选择模式
    def on_mode_changed(self, text):
        self.ui.stackedWidget_Mode.setCurrentIndex(self.ui.comboBox_mode.currentIndex())
        if self.ui.stackedWidget_Mode.currentIndex() == 0:
            CONTROL.TESTING = False
            self.predictor_result.trig_stop = 1
            self.ui.label_result.setText('')
            self.ui.label_action.setText('')
            self.DrawPlot.close()

        if self.ui.stackedWidget_Mode.currentIndex() == 1:
            CONTROL.TESTING = True
            self.predictor_result.trig_stop = 0

            t = threading.Thread(target=self.predictor_result.get_action, )
            t.daemon = True
            t.start()

            self.ui.label_result.setText('')
            self.ui.label_action.setText('')
            self.DrawPlot.show()

        if self.ui.stackedWidget_Mode.currentIndex() == 2:
            CONTROL.TESTING = False
            self.predictor_result.trig_stop = 1
            self.ui.label_result.setText('')
            self.ui.label_action.setText('')
            self.DrawPlot.close()

    def setLowPos(self):
        if  abs(self.__last_changed_height - self.tableController.getHeight()) > 0.1 and not (self.tableController.getTableMovingStatus()) and self.tableController.targetStatus == -1:
            print("设置低位")
            self.__last_changed_height = self.tableController.getHeight()
            self.tableController.targetStatus = 0
            self.lastSendHeight = 0

    def setHighPos(self):
        if  abs(self.__last_changed_height - self.tableController.getHeight()) > 0.1 and not (self.tableController.getTableMovingStatus()) and self.tableController.targetStatus == -1:
            print("设置高位")
            self.__last_changed_height = self.tableController.getHeight()
            self.tableController.targetStatus = 2
            self.lastSendHeight = 2

    # 选择姿态
    def on_chooseposture_changed(self, text):
        self.setFocus()

    # 选择人员
    def on_chooseperson_changed(self, text):
        self.setFocus()

    # 选择高低位
    def on_chooseposition_changed(self, text):
        self.setFocus()

    def arrays_equal(self, arr1, arr2):
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
        if self.ui.stackedWidget_Mode.currentIndex() == 0:  # 采集模式
            names = self.getnames('./persondata')
            current_person = self.ui.comboBox_chooseperson.currentText()
            if not self.arrays_equal(names, self.allname):
                print("存在姓名更改")
                self.ui.comboBox_chooseperson.clear()
                for per_name in names:
                    self.ui.comboBox_chooseperson.addItem(per_name)
                if current_person in names:
                    self.ui.comboBox_chooseperson.setCurrentText(current_person)
            self.allname = names

    def refresh_tableheight_timer_callback(self):
        if self.ui.stackedWidget_Mode.currentIndex() == 1:  # 推理模式下显示升降桌高度
            if self.tableController is not None:
                if self.tableDataActivate == False:
                    self.ui.label_table_current_height.setText("未激活")
                    self.ui.comboBox_predictPos.setCurrentText("未知")
                if 90.0 > self.tableController.current_height >= 10.0:
                    self.ui.label_table_current_height.setText(format(self.tableController.current_height, ".1f"))
                    self.ui.comboBox_predictPos.setCurrentText("低位")
                    self.tableDataActivate = True
                elif self.tableController.current_height > 90.0:
                    self.ui.label_table_current_height.setText(format(self.tableController.current_height, ".1f"))
                    self.ui.comboBox_predictPos.setCurrentText("高位")
                    self.tableDataActivate = True
                else:
                    pass



    # 检测串口
    def click_pushButton_checkcoms(self):
        self.ui.comboBox_Coms.clear()
        self.com_info.clear()
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            location_com = port.name
            self.ui.comboBox_Coms.addItem(f"{location_com}")
            self.com_info[location_com] = port.device

    def click_pushButton_setHeight(self):
        toTargetHeight = self.ui.spinBox_setHeight.value()
        self.tableController.setHeight(toTargetHeight)

    # 快速打开保存文件夹
    def click_pushButton_quickopen(self):
        start_directory = pl.Path('./persondata')
        os.system("explorer.exe %s" % start_directory)

    # 显示数据
    def click_pushButton_show(self):
        if self.iscomdata != 1:
            ConnectDiaglog = ConnectDeviceDialog()
            items = [self.ui.comboBox_Coms.itemText(i) for i in range(self.ui.comboBox_Coms.count())]
            ConnectDiaglog.ui.comboBox_device.addItems(items)
            ConnectDiaglog.ui.comboBox_table.addItems(items)
            result = ConnectDiaglog.exec()

            # 处理用户的选择
            if result == QtWidgets.QDialog.Accepted:
                self.com_device = self.com_info[ConnectDiaglog.ui.comboBox_device.currentText()]
                self.com_table = self.com_info[ConnectDiaglog.ui.comboBox_table.currentText()]
            else:
                return

            try:
                # 连接检测设备
                parser = ArgumentParser()
                parser.add_argument('--port', type=int, default=SOCKET.SERVER_PORT)
                parser.add_argument('--serial', type=str, default=self.com_device)
                print("检测设备串口:", self.com_device)
                args = parser.parse_args()
                SOCKET.SERVER_PORT = args.port
                self.myserial = MySerial_2head1tail(b'\xFA', args.serial, b'\xAF', b'\xFF', length=[64 * 4 + 1, 5, 21])
                self.myserial.start_report()
                print("检测连接成功:")
                # # 连接升降桌
                # self.tableController.startCtrl(port=self.com_table)
                # print("升降桌连接成功:")
                # print("升降桌串口:", self.com_table)
            except Exception as e:
                QMessageBox.information(self, "提示", "设备串口连接失败！")
                return

            self.ui.pushButton_checkcoms.setEnabled(0)
            self.ui.comboBox_Coms.setEnabled(0)
            self.ui.pushButton_show.setEnabled(0)

            jobs = [self.myserial.message_classify, self.sonic_data_collector.play_sonic,
                    self.ir_data_collector.play_IR,
                    self.predictor_result.get_action]
            for job in jobs:
                t = threading.Thread(target=job, )
                t.daemon = True
                t.start()
                self.workThreads.append(t)

            self.iscomdata = 1
            self.predictor_result.filter_mode_on()

    # 新增人员
    def clickpushButton_add(self):
        print(self.size())
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

    def getnames(self, directory):
        folders = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                folders.append(item)
        return folders

    def get_savepath_personal(self, filename):
        button = QMessageBox.question(self, "提示", "是否保存")
        if button == QMessageBox.StandardButton.Yes:
            person_name = self.ui.comboBox_chooseperson.currentText()
            table_pos = self.ui.comboBox_chooseposition.currentText()
            person_pos = self.ui.comboBox_chooseposture.currentText()
            sceneroot = pl.Path('./persondata') / pl.Path('./' + person_name) / pl.Path('./' + table_pos) / pl.Path(
                './' + person_pos)

            os.makedirs(sceneroot, exist_ok=True)
            return [1, filename, sceneroot]
        else:
            return False

    def get_savepath_training(self, filename):
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

        trainroot = pl.Path('./data') / pl.Path('./' + save_path2)
        return [1, filename, trainroot]

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
                file_name = "new" + person_name + "_" + table_pos + "_" + person_pos + "_" + f'{timestamp}'

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
        # if distance >= 199.000:
        #     self.ui.label_Sonic.setText("超声数据：" + " " * 28 + '-----')
        # else:
        self.ui.label_Sonic.setText("超声数据：" + " " * 24 + '%9.3f' % distance)

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

    def update_predict_result(self, result):
        if result[0] == 'idle':
            self.ui.label_result.setText('无人')
        elif result[0] == 'sit':
            self.ui.label_result.setText('坐姿')
        elif result[0] == 'stand':
            self.ui.label_result.setText('站姿')
        elif result[0] == 'sit2stand':
            self.ui.label_result.setText('坐姿---站姿')
        elif result[0] == 'not sure':
            self.ui.label_result.setText('不确定')
        elif result[0] == 'stand2sit':
            self.ui.label_result.setText('站姿---坐姿')

        self.ui.label_action.setText(result[1])

        pose_probility = result[2][0]
        self.DrawPlot.updateData(pose_probility)
        self.DrawPlot.update()

        if result[1] == "升起":
            self.setHighPos()
            if self.lastResult == result[1]:
                self.predictequalTimes += 1
            else:
                self.predictequalTimes = 0
            if self.predictequalTimes == 2:
                self.predictequalTimes = 0

        elif result[1] == "下降":
            self.setLowPos()
            if self.lastResult == result[1]:
                self.predictequalTimes += 1
            else:
                self.predictequalTimes = 0
            if self.predictequalTimes == 2:
                self.predictequalTimes = 0

        else:
            pass
        self.lastResult = result[1]

    # IRDataCollect数据传输
    def update_heatmap_display(self, heatmap):
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
