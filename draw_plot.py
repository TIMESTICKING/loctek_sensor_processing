import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtChart import QChart, QChartView, QLineSeries,QBarCategoryAxis,QValueAxis
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt


class DrawPlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("推理结果概率")
        self.setFixedSize(800, 600)  # Set fixed size

        self.setupChart()


    def setupChart(self):
        # Create a line series
        self.series = QLineSeries()

        # Create a chart and add the line series
        self.chart = QChart()
        self.chart.addSeries(self.series)

        # Set title and axes labels
        self.chart.setTitle("姿态推理概率密度:")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)

        categories = ['无人', '坐姿', '坐——站', '站姿', '站——坐']
        axisX = QBarCategoryAxis()
        axisX.append(categories)
        self.chart.addAxis(axisX, Qt.AlignBottom)
        self.series.attachAxis(axisX)

        axisY = QValueAxis()
        axisY.setRange(0, 1)
        self.chart.addAxis(axisY, Qt.AlignLeft)
        self.series.attachAxis(axisY)

        # Create a chart view and set the chart
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)

        self.chartView = QChartView(self.chart)
        self.chartView.setRenderHint(QPainter.Antialiasing)

        layout = QVBoxLayout()
        layout.addWidget(self.chartView)
        self.setLayout(layout)

    # def setupUI(self):
    #     self.updateButton = QPushButton("Update Data", self)
    #     self.updateButton.clicked.connect(self.updateData)

    def updateData(self,new_y_values):
        # Define new data points
        new_x_values = [0, 1, 2, 3, 4]
        # Clear existing data
        self.series.clear()
        # Add new data points to the series
        for x, y in zip(new_x_values, new_y_values):
            self.series.append(x, y)


