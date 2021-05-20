# -*-coding:utf-8-*-
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import numpy as np
from ui4 import Ui_form
import pyqtgraph as pg


class MainDialogImgBW(QMainWindow, Ui_form):
    def __init__(self):
        super(MainDialogImgBW, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Sleep Stage Scoring")
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.i = 0
        self.N = 200
        self.stage = 4
        self.dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.class_dict = {0: 'W', 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.F = pg.GraphicsLayoutWidget()  # 定义一个GraphicsLayoutWidget 对象
        self.gridlayout.addWidget(self.F)  # 将对象添加到容器中去

        self.gridlayout_2 = QGridLayout(self.groupBox_4)  # 继承容器groupBox_4
        self.FFT = pg.GraphicsLayoutWidget()  # 定义一个GraphicsLayoutWidget 对象
        self.gridlayout_2.addWidget(self.FFT)  # 将对象添加到容器中去

        self.gridlayout_3 = QGridLayout(self.groupBox_5)  # 继承容器groupBox_5
        self.RW = pg.GraphicsLayoutWidget()  # 定义一个GraphicsLayoutWidget 对象
        self.gridlayout_3.addWidget(self.RW)  # 将对象添加到容器中去

        self.init_F()
        self.init_RW()
        self.init_FFT()

        data = np.load("./eeg_data.npz", allow_pickle=True)
        self.x = data['x']
        self.y = data['y']
        self.sampling_rate = int(data['fs'])

        self.init_timer()

    def init_timer(self):
        self.timer_re = QTimer(self)
        self.timer_re.start(10)  # ms
        self.timer_re.timeout.connect(self.plotData)

    def init_F(self):
        set_left = pg.AxisItem('left', pen='y',  maxTickLength=-5)
        set_bottom = pg.AxisItem('bottom', pen='y',  maxTickLength=-5)
        p = self.F.addPlot(row=0, col=0, axisItems={'left': set_left, 'bottom': set_bottom})  # 新建一个子图
        # p2 = self.F.addPlot(row=1, col=0)
        # p.showGrid(x=True, y=True)
        p.setRange(yRange=(-100, 100), xRange=(10, 1000), disableAutoRange=True)  # 设置坐标轴刻度范围
        p.addLine(y=0, pen=(255, 255, 0))  # label='Frequence(Hz)'
        p.hideAxis('bottom')
        p.setLabel(axis='left', text='Amplitude(uV)', color='yellow')
        # p.setLabel(axis='bottom', text='Frequence(Hz)', color='yellow')
        # p.setTitle('EEG', color='r', fontsize=20,)
        p.addLegend(offset=(10, 10))  # 添加label,并设置位置
        self.curve1 = p.plot(pen="r", name="channel 1")

    def init_RW(self):
        set_left = pg.AxisItem('left', pen='y',  maxTickLength=-5)
        set_bottom = pg.AxisItem('bottom', pen='y',maxTickLength=-5)
        p2 = self.RW.addPlot(row=0, col=0, axisItems={'left': set_left, 'bottom': set_bottom})  # 新建一个子图
        # p2.showGrid(x=True, y=True)
        p2.setRange(yRange=(-100, 100), xRange=(10, 1000), disableAutoRange=True)  # 设置坐标轴刻度范围
        p2.addLine(y=0, pen=(255, 255, 0))
        p2.hideAxis('bottom')
        p2.setLabel(axis='left', text='Amplitude(uV)', color='yellow')
        # p.setLabel(axis='bottom', text='Frequence(Hz)', color='yellow')
        # p.setTitle('EEG', color='r', fontsize=20,)
        p2.addLegend(offset=(10, 10))  # 添加label,并设置位置
        self.curve3 = p2.plot(pen="r", name="seta")
        self.curve4 = p2.plot(pen='b', name='alpha')

    def init_FFT(self):
        set_left = pg.AxisItem('left', pen='y',  maxTickLength=-5)
        set_bottom = pg.AxisItem('bottom', pen='y',maxTickLength=-5)
        p3 = self.FFT.addPlot(row=0, col=0, axisItems={'left': set_left, 'bottom': set_bottom})  # 新建一个子图
        # p2.showGrid(x=True, y=True)
        p3.setRange(yRange=(-0, 10), xRange=(0, 100), disableAutoRange=True)  # 设置坐标轴刻度范围
        # p3.setLabel(axis='left', text='Amplitude(uV)', color='yellow')
        # p3.setLabel(axis='bottom', text='Frequence(Hz)', color='yellow')
        # p.setTitle('EEG', color='r', fontsize=20,)
        p3.addLegend(offset=(10, 10))  # 添加label,并设置位置
        self.curve5 = p3.plot(pen="r", name="channel 1")

    def plotData(self):
        data_x = self.x[self.i * 1: 400 + self.i * 1]
        self.curve1.setData(data_x[::-1])
        self.curve3.setData(data_x)
        self.curve4.setData(data_x[::-1])

        x = self.N / 2 * np.linspace(0, 1, self.N / 2)
        transformed = 1.0 / self.N * np.abs(np.fft.fft(data_x, self.N))
        self.curve5.setData(x, transformed[0:int(self.N / 2)], )

        label = (self.i + 10) // 300
        self.stage = self.y[label]
        if (self.i + 10) % 300 == 0:
            self.dict[self.stage] = self.dict[self.stage] + 1
        # 显示分期结果
        self.classes(self.stage, self.sampling_rate)
        self.i = self.i + 1

    def classes(self, stage, sampling_rate):
        self.lineEdit.setText(str(self.dict[0]))
        self.lineEdit_2.setText(str(self.dict[1]))
        self.lineEdit_3.setText(str(self.dict[2]))
        self.lineEdit_4.setText(str(self.dict[3]))
        self.lineEdit_5.setText(str(self.dict[4]))
        self.lineEdit_6.setText(self.class_dict[stage])

        self.lineEdit_7.setText(str(sampling_rate))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    # main.showFullScreen()
    app.installEventFilter(main)
    sys.exit(app.exec_())
