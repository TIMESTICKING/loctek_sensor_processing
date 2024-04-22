# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'serialShow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SerialShow(object):
    def setupUi(self, SerialShow):
        SerialShow.setObjectName("SerialShow")
        SerialShow.resize(790, 446)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SerialShow.sizePolicy().hasHeightForWidth())
        SerialShow.setSizePolicy(sizePolicy)
        SerialShow.setMinimumSize(QtCore.QSize(0, 0))
        SerialShow.setMaximumSize(QtCore.QSize(10000, 10000))
        SerialShow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(SerialShow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget_4 = QtWidgets.QWidget(self.centralwidget)
        self.widget_4.setStyleSheet("")
        self.widget_4.setObjectName("widget_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.widget_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.widget_9 = QtWidgets.QWidget(self.widget_4)
        self.widget_9.setObjectName("widget_9")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.widget_9)
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8.setSpacing(0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.comboBox_mode = QtWidgets.QComboBox(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_mode.sizePolicy().hasHeightForWidth())
        self.comboBox_mode.setSizePolicy(sizePolicy)
        self.comboBox_mode.setMinimumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_mode.setFont(font)
        self.comboBox_mode.setObjectName("comboBox_mode")
        self.comboBox_mode.addItem("")
        self.comboBox_mode.addItem("")
        self.gridLayout_8.addWidget(self.comboBox_mode, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget_9)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_8.addWidget(self.label_5, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_8.addItem(spacerItem, 0, 2, 1, 1)
        self.gridLayout_7.addWidget(self.widget_9, 0, 0, 1, 1)
        self.stackedWidget_Mode = QtWidgets.QStackedWidget(self.widget_4)
        self.stackedWidget_Mode.setObjectName("stackedWidget_Mode")
        self.stackedWidget_4Page1 = QtWidgets.QWidget()
        self.stackedWidget_4Page1.setObjectName("stackedWidget_4Page1")
        self.gridLayout = QtWidgets.QGridLayout(self.stackedWidget_4Page1)
        self.gridLayout.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_7 = QtWidgets.QWidget(self.stackedWidget_4Page1)
        self.widget_7.setObjectName("widget_7")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget_7)
        self.gridLayout_5.setContentsMargins(10, 0, 10, 0)
        self.gridLayout_5.setHorizontalSpacing(10)
        self.gridLayout_5.setVerticalSpacing(0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.pushButton_quickopen = QtWidgets.QPushButton(self.widget_7)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_quickopen.setFont(font)
        self.pushButton_quickopen.setObjectName("pushButton_quickopen")
        self.gridLayout_5.addWidget(self.pushButton_quickopen, 0, 0, 1, 1)
        self.pushButton_addperson = QtWidgets.QPushButton(self.widget_7)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_addperson.setFont(font)
        self.pushButton_addperson.setObjectName("pushButton_addperson")
        self.gridLayout_5.addWidget(self.pushButton_addperson, 0, 1, 1, 1)
        self.gridLayout_5.setColumnStretch(0, 1)
        self.gridLayout_5.setColumnStretch(1, 1)
        self.gridLayout.addWidget(self.widget_7, 0, 0, 1, 1)
        self.widget_5 = QtWidgets.QWidget(self.stackedWidget_4Page1)
        self.widget_5.setStyleSheet("")
        self.widget_5.setObjectName("widget_5")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget_5)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_2 = QtWidgets.QLabel(self.widget_5)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)
        self.comboBox_chooseperson = QtWidgets.QComboBox(self.widget_5)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_chooseperson.setFont(font)
        self.comboBox_chooseperson.setObjectName("comboBox_chooseperson")
        self.gridLayout_3.addWidget(self.comboBox_chooseperson, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget_5)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.comboBox_chooseposition = QtWidgets.QComboBox(self.widget_5)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_chooseposition.setFont(font)
        self.comboBox_chooseposition.setObjectName("comboBox_chooseposition")
        self.gridLayout_3.addWidget(self.comboBox_chooseposition, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.widget_5)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 2, 0, 1, 1)
        self.comboBox_chooseposture = QtWidgets.QComboBox(self.widget_5)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_chooseposture.setFont(font)
        self.comboBox_chooseposture.setObjectName("comboBox_chooseposture")
        self.gridLayout_3.addWidget(self.comboBox_chooseposture, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.widget_5, 1, 0, 1, 1)
        self.widget_6 = QtWidgets.QWidget(self.stackedWidget_4Page1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_6.sizePolicy().hasHeightForWidth())
        self.widget_6.setSizePolicy(sizePolicy)
        self.widget_6.setObjectName("widget_6")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget_6)
        self.gridLayout_4.setContentsMargins(-1, 0, 0, -1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pushButton_start = QtWidgets.QPushButton(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_start.sizePolicy().hasHeightForWidth())
        self.pushButton_start.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.gridLayout_4.addWidget(self.pushButton_start, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.widget_6, 2, 0, 1, 1)
        self.gridLayout.setRowMinimumHeight(0, 1)
        self.gridLayout.setRowMinimumHeight(1, 2)
        self.gridLayout.setRowMinimumHeight(2, 3)
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 6)
        self.gridLayout.setRowStretch(2, 1)
        self.stackedWidget_Mode.addWidget(self.stackedWidget_4Page1)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_9.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_9.setSpacing(0)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.widget_10 = QtWidgets.QWidget(self.page)
        self.widget_10.setObjectName("widget_10")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.widget_10)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_6 = QtWidgets.QLabel(self.widget_10)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_10.addWidget(self.label_6, 1, 0, 1, 1)
        self.comboBox_predictPos = QtWidgets.QComboBox(self.widget_10)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_predictPos.setFont(font)
        self.comboBox_predictPos.setObjectName("comboBox_predictPos")
        self.comboBox_predictPos.addItem("")
        self.comboBox_predictPos.addItem("")
        self.gridLayout_10.addWidget(self.comboBox_predictPos, 1, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.widget_10)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_10.addWidget(self.label_7, 0, 0, 1, 1)
        self.label_table_current_height = QtWidgets.QLabel(self.widget_10)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_table_current_height.setFont(font)
        self.label_table_current_height.setText("")
        self.label_table_current_height.setObjectName("label_table_current_height")
        self.gridLayout_10.addWidget(self.label_table_current_height, 0, 1, 1, 1)
        self.gridLayout_9.addWidget(self.widget_10, 0, 0, 1, 1)
        self.widget_11 = QtWidgets.QWidget(self.page)
        self.widget_11.setObjectName("widget_11")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.widget_11)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_result = QtWidgets.QLabel(self.widget_11)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_result.setFont(font)
        self.label_result.setText("")
        self.label_result.setObjectName("label_result")
        self.gridLayout_11.addWidget(self.label_result, 0, 1, 1, 1)
        self.label_action = QtWidgets.QLabel(self.widget_11)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_action.setFont(font)
        self.label_action.setText("")
        self.label_action.setObjectName("label_action")
        self.gridLayout_11.addWidget(self.label_action, 1, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.widget_11)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_11.addWidget(self.label_10, 1, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.widget_11)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.gridLayout_11.addWidget(self.label_9, 0, 0, 1, 1)
        self.gridLayout_9.addWidget(self.widget_11, 1, 0, 1, 1)
        self.gridLayout_9.setRowMinimumHeight(0, 1)
        self.gridLayout_9.setRowMinimumHeight(1, 1)
        self.gridLayout_9.setRowStretch(0, 1)
        self.gridLayout_9.setRowStretch(1, 2)
        self.stackedWidget_Mode.addWidget(self.page)
        self.gridLayout_7.addWidget(self.stackedWidget_Mode, 1, 0, 1, 1)
        self.gridLayout_7.setRowStretch(0, 1)
        self.gridLayout_7.setRowStretch(1, 10)
        self.horizontalLayout_2.addWidget(self.widget_4)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_2.addWidget(self.line)
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setMinimumSize(QtCore.QSize(400, 0))
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_8 = QtWidgets.QWidget(self.widget_2)
        self.widget_8.setObjectName("widget_8")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.widget_8)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.pushButton_checkcoms = QtWidgets.QPushButton(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_checkcoms.sizePolicy().hasHeightForWidth())
        self.pushButton_checkcoms.setSizePolicy(sizePolicy)
        self.pushButton_checkcoms.setMinimumSize(QtCore.QSize(0, 34))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_checkcoms.setFont(font)
        self.pushButton_checkcoms.setObjectName("pushButton_checkcoms")
        self.gridLayout_12.addWidget(self.pushButton_checkcoms, 0, 0, 1, 1)
        self.comboBox_Coms = QtWidgets.QComboBox(self.widget_8)
        self.comboBox_Coms.setMinimumSize(QtCore.QSize(0, 34))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_Coms.setFont(font)
        self.comboBox_Coms.setObjectName("comboBox_Coms")
        self.gridLayout_12.addWidget(self.comboBox_Coms, 0, 1, 1, 1)
        self.pushButton_show = QtWidgets.QPushButton(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_show.sizePolicy().hasHeightForWidth())
        self.pushButton_show.setSizePolicy(sizePolicy)
        self.pushButton_show.setMinimumSize(QtCore.QSize(0, 34))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_show.setFont(font)
        self.pushButton_show.setObjectName("pushButton_show")
        self.gridLayout_12.addWidget(self.pushButton_show, 0, 2, 1, 1)
        self.verticalLayout.addWidget(self.widget_8)
        self.widget = QtWidgets.QWidget(self.widget_2)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.IR_showLabel = QtWidgets.QLabel(self.widget)
        self.IR_showLabel.setText("")
        self.IR_showLabel.setObjectName("IR_showLabel")
        self.horizontalLayout.addWidget(self.IR_showLabel)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.verticalLayout.addWidget(self.widget)
        self.widget_3 = QtWidgets.QWidget(self.widget_2)
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_Sonic = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_Sonic.setFont(font)
        self.label_Sonic.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_Sonic.setObjectName("label_Sonic")
        self.gridLayout_2.addWidget(self.label_Sonic, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.widget_3)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout_2.addWidget(self.widget_2)
        SerialShow.setCentralWidget(self.centralwidget)

        self.retranslateUi(SerialShow)
        self.stackedWidget_Mode.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(SerialShow)

    def retranslateUi(self, SerialShow):
        _translate = QtCore.QCoreApplication.translate
        SerialShow.setWindowTitle(_translate("SerialShow", "红外+超声数据采集"))
        self.comboBox_mode.setItemText(0, _translate("SerialShow", "采集模式"))
        self.comboBox_mode.setItemText(1, _translate("SerialShow", "推理模式"))
        self.label_5.setText(_translate("SerialShow", "选择模式："))
        self.pushButton_quickopen.setText(_translate("SerialShow", "快速打开保存文件夹"))
        self.pushButton_addperson.setText(_translate("SerialShow", "新增人员"))
        self.label_2.setText(_translate("SerialShow", "选择人员:"))
        self.label_4.setText(_translate("SerialShow", "选择高低位："))
        self.label.setText(_translate("SerialShow", "选择姿态："))
        self.pushButton_start.setText(_translate("SerialShow", "开始记录"))
        self.label_6.setText(_translate("SerialShow", "升降桌位置:"))
        self.comboBox_predictPos.setItemText(0, _translate("SerialShow", "低位"))
        self.comboBox_predictPos.setItemText(1, _translate("SerialShow", "高位"))
        self.label_7.setText(_translate("SerialShow", "升降桌高度:"))
        self.label_10.setText(_translate("SerialShow", "推理桌子动作："))
        self.label_9.setText(_translate("SerialShow", "推理结果："))
        self.pushButton_checkcoms.setText(_translate("SerialShow", "检测串口"))
        self.pushButton_show.setText(_translate("SerialShow", "连接设备"))
        self.label_3.setText(_translate("SerialShow", "红外数据:"))
        self.label_Sonic.setText(_translate("SerialShow", "超声数据："))
