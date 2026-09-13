# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QWidget, QHeaderView, QTableWidgetItem, QAbstractItemView, \
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QLineEdit, \
    QComboBox, QCheckBox, QDoubleSpinBox, QFrame, QSplitter, QTableWidget, \
    QGridLayout, QSizePolicy, QFormLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QFont, QPalette, QColor
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
sys.path.append('UIProgram')
import sys
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
import torch


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("基于深度学习的交通标志与行人车辆检测系统")
        self.setMinimumSize(1500, 900)
        
        self.init_params()
        self.setup_ui()
        self.connect_signals()
        
        screen = QApplication.primaryScreen().geometry()
        self.resize(int(screen.width() * 0.85), int(screen.height() * 0.85))
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

    def init_params(self):
        self.show_width = 640
        self.show_height = 480
        self.org_path = None
        self.is_camera_open = False
        self.cap = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.show_labels = True
        
        self.set_fonts()
        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        try:
            self.model = YOLO(Config.model_path, task='detect')
            self.model(np.zeros((48, 48, 3)), device=self.device)
        except:
            print("模型加载失败，请检查模型路径")
            self.model = None
        
        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)
        self.colors = tools.Colors()
        self.timer_camera = QTimer()
        
        self.location_list = []
        self.cls_list = []
        self.conf_list = []
        self.results = None
        self.org_img = None
        self.draw_img = None
        self.img_width = 640
        self.img_height = 480

    def set_fonts(self):
        self.title_font = QFont("Microsoft YaHei", 16, QFont.Bold)
        self.group_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        self.label_font = QFont("Microsoft YaHei", 11)
        self.value_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        self.button_font = QFont("Microsoft YaHei", 11, QFont.Bold)
        self.table_font = QFont("Microsoft YaHei", 11)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 标题栏
        self.setup_title_bar(main_layout)
        
        # 中间区域：左右分割
        center_splitter = QSplitter(Qt.Horizontal)
        
        # ==================== 左侧图像显示区 ====================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.label_show = QLabel()
        self.label_show.setMinimumSize(800, 500)
        self.label_show.setAlignment(Qt.AlignCenter)
        self.label_show.setStyleSheet("""
            QLabel {
                border: 2px solid #409EFF;
                background: #1a1a2e;
                border-radius: 8px;
            }
        """)
        self.label_show.setScaledContents(True)
        left_layout.addWidget(self.label_show)
        
        # ==================== 右侧控制区 ====================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 检测参数设置
        self.create_param_group(right_layout)
        
        # 检测结果指标卡（扩大版）
        self.create_metric_group(right_layout)
        
        # 操作按钮（多色按钮）
        self.create_button_group(right_layout)
        
        # 数据源
        self.create_source_group(right_layout)
        
        right_layout.addStretch()
        
        center_splitter.addWidget(left_widget)
        center_splitter.addWidget(right_widget)
        center_splitter.setSizes([900, 500])
        
        main_layout.addWidget(center_splitter)
        
        # 表格区域
        self.create_table_group(main_layout)

    def setup_title_bar(self, parent_layout):
        title_widget = QWidget()
        title_widget.setFixedHeight(55)
        title_widget.setStyleSheet("""
            QWidget {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f4c81, stop:1 #409EFF);
                border-radius: 6px;
            }
        """)
        
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(15, 0, 15, 0)
        
        title_label = QLabel("基于深度学习的交通标志与行人车辆检测系统")
        title_label.setFont(self.title_font)
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(title_label)
        parent_layout.addWidget(title_widget)

    def create_param_group(self, parent_layout):
        """检测参数设置组"""
        self.groupBox = QGroupBox("检测参数设置")
        self.groupBox.setFont(self.group_font)
        self.groupBox.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
        """)
        
        layout = QHBoxLayout(self.groupBox)
        layout.setSpacing(25)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值：")
        conf_label.setFont(self.label_font)
        self.doubleSpinBox = QDoubleSpinBox()
        self.doubleSpinBox.setRange(0.0, 1.0)
        self.doubleSpinBox.setSingleStep(0.05)
        self.doubleSpinBox.setValue(self.conf_thres)
        self.doubleSpinBox.setFont(self.value_font)
        self.doubleSpinBox.setFixedWidth(100)
        self.doubleSpinBox.setFixedHeight(34)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.doubleSpinBox)
        
        # IOU阈值
        iou_layout = QHBoxLayout()
        iou_label = QLabel("交并比阈值：")
        iou_label.setFont(self.label_font)
        self.doubleSpinBox_2 = QDoubleSpinBox()
        self.doubleSpinBox_2.setRange(0.0, 1.0)
        self.doubleSpinBox_2.setSingleStep(0.05)
        self.doubleSpinBox_2.setValue(self.iou_thres)
        self.doubleSpinBox_2.setFont(self.value_font)
        self.doubleSpinBox_2.setFixedWidth(100)
        self.doubleSpinBox_2.setFixedHeight(34)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.doubleSpinBox_2)
        
        self.checkBox = QCheckBox("显示标签名称与置信度")
        self.checkBox.setChecked(self.show_labels)
        self.checkBox.setFont(self.label_font)
        
        layout.addLayout(conf_layout)
        layout.addLayout(iou_layout)
        layout.addWidget(self.checkBox)
        layout.addStretch()
        
        parent_layout.addWidget(self.groupBox)

    def create_metric_group(self, parent_layout):
        """检测结果指标卡 - 扩大版，4个卡片均匀分布"""
        self.groupBox_2 = QGroupBox("检测结果")
        self.groupBox_2.setFont(self.group_font)
        self.groupBox_2.setMinimumHeight(170)
        self.groupBox_2.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
        """)
        
        layout = QHBoxLayout(self.groupBox_2)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 卡片1：检测目标数
        card1 = QFrame()
        card1.setStyleSheet("background-color: #ecf5ff; border-radius: 8px;")
        card1_layout = QVBoxLayout(card1)
        card1_layout.setAlignment(Qt.AlignCenter)
        card1_layout.setContentsMargins(15, 12, 15, 12)
        
        self.metric_targets = QLabel("0")
        self.metric_targets.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        self.metric_targets.setStyleSheet("color: #409EFF;")
        self.metric_targets.setAlignment(Qt.AlignCenter)
        
        targets_label = QLabel("检测目标")
        targets_label.setFont(self.label_font)
        targets_label.setStyleSheet("color: #909399;")
        targets_label.setAlignment(Qt.AlignCenter)
        
        card1_layout.addWidget(self.metric_targets)
        card1_layout.addWidget(targets_label)
        
        # 卡片2：平均置信度
        card2 = QFrame()
        card2.setStyleSheet("background-color: #f0f9eb; border-radius: 8px;")
        card2_layout = QVBoxLayout(card2)
        card2_layout.setAlignment(Qt.AlignCenter)
        card2_layout.setContentsMargins(15, 12, 15, 12)
        
        self.metric_conf = QLabel("0%")
        self.metric_conf.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        self.metric_conf.setStyleSheet("color: #67C23A;")
        self.metric_conf.setAlignment(Qt.AlignCenter)
        
        conf_label = QLabel("平均置信度")
        conf_label.setFont(self.label_font)
        conf_label.setStyleSheet("color: #909399;")
        conf_label.setAlignment(Qt.AlignCenter)
        
        card2_layout.addWidget(self.metric_conf)
        card2_layout.addWidget(conf_label)
        
        # 卡片3：推理耗时
        card3 = QFrame()
        card3.setStyleSheet("background-color: #fdf6ec; border-radius: 8px;")
        card3_layout = QVBoxLayout(card3)
        card3_layout.setAlignment(Qt.AlignCenter)
        card3_layout.setContentsMargins(15, 12, 15, 12)
        
        self.metric_time = QLabel("0s")
        self.metric_time.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        self.metric_time.setStyleSheet("color: #E6A23C;")
        self.metric_time.setAlignment(Qt.AlignCenter)
        
        time_label = QLabel("推理耗时")
        time_label.setFont(self.label_font)
        time_label.setStyleSheet("color: #909399;")
        time_label.setAlignment(Qt.AlignCenter)
        
        card3_layout.addWidget(self.metric_time)
        card3_layout.addWidget(time_label)
        
        # 卡片4：当前选中目标
        card4 = QFrame()
        card4.setStyleSheet("background-color: #f5f7fa; border-radius: 8px;")
        card4_layout = QVBoxLayout(card4)
        card4_layout.setContentsMargins(12, 10, 12, 10)
        
        self.current_type_label = QLabel("类型：--")
        self.current_type_label.setFont(self.label_font)
        self.current_type_label.setStyleSheet("color: #606266;")
        
        self.current_conf_label = QLabel("置信度：--")
        self.current_conf_label.setFont(self.label_font)
        self.current_conf_label.setStyleSheet("color: #606266;")
        
        self.current_pos_label = QLabel("位置：[--]")
        self.current_pos_label.setFont(self.label_font)
        self.current_pos_label.setStyleSheet("color: #606266;")
        
        card4_layout.addWidget(self.current_type_label)
        card4_layout.addWidget(self.current_conf_label)
        card4_layout.addWidget(self.current_pos_label)
        
        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addWidget(card3)
        layout.addWidget(card4)
        layout.addStretch()
        
        parent_layout.addWidget(self.groupBox_2)

    def create_button_group(self, parent_layout):
        """操作按钮 - 多色按钮"""
        self.groupBox_4 = QGroupBox("操作")
        self.groupBox_4.setFont(self.group_font)
        self.groupBox_4.setMinimumHeight(120)
        self.groupBox_4.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
        """)
        
        layout = QGridLayout(self.groupBox_4)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 12, 15, 12)
        
        # 主按钮样式（蓝色）
        primary_btn_style = """
            QPushButton {
                background: #409EFF;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                min-height: 42px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #66b1ff;
            }
        """
        
        # 次按钮样式（绿色）
        success_btn_style = """
            QPushButton {
                background: #67C23A;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                min-height: 42px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #85ce61;
            }
        """
        
        # 危险按钮样式（红色）
        danger_btn_style = """
            QPushButton {
                background: #F56C6C;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                min-height: 42px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #f78989;
            }
        """
        
        # 创建按钮
        self.PicBtn = QPushButton("📁 打开图片")
        self.PicBtn.setStyleSheet(primary_btn_style)
        
        self.VideoBtn = QPushButton("🎬 打开视频")
        self.VideoBtn.setStyleSheet(primary_btn_style)
        
        self.CapBtn = QPushButton("📷 打开摄像头")
        self.CapBtn.setStyleSheet(primary_btn_style)
        
        self.FilesBtn = QPushButton("📂 批量检测")
        self.FilesBtn.setStyleSheet(success_btn_style)
        
        self.SaveBtn = QPushButton("💾 保存结果")
        self.SaveBtn.setStyleSheet(success_btn_style)
        
        self.ExitBtn = QPushButton("🚪 退出系统")
        self.ExitBtn.setStyleSheet(danger_btn_style)
        
        # 布局：第一行3个主按钮，第二行2个次按钮+1个危险按钮
        layout.addWidget(self.PicBtn, 0, 0)
        layout.addWidget(self.VideoBtn, 0, 1)
        layout.addWidget(self.CapBtn, 0, 2)
        layout.addWidget(self.FilesBtn, 1, 0)
        layout.addWidget(self.SaveBtn, 1, 1)
        layout.addWidget(self.ExitBtn, 1, 2)
        
        parent_layout.addWidget(self.groupBox_4)

    def create_source_group(self, parent_layout):
        """数据源组"""
        self.source_group = QGroupBox("数据源")
        self.source_group.setFont(self.group_font)
        self.source_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
        """)
        
        layout = QFormLayout(self.source_group)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 12, 15, 12)
        
        self.PiclineEdit = QLineEdit()
        self.PiclineEdit.setPlaceholderText("未选择图片")
        self.PiclineEdit.setReadOnly(True)
        self.PiclineEdit.setFixedHeight(36)
        self.PiclineEdit.setStyleSheet("border: 1px solid #dcdfe6; border-radius: 4px; padding: 6px;")
        
        self.VideolineEdit = QLineEdit()
        self.VideolineEdit.setPlaceholderText("未选择视频")
        self.VideolineEdit.setReadOnly(True)
        self.VideolineEdit.setFixedHeight(36)
        self.VideolineEdit.setStyleSheet("border: 1px solid #dcdfe6; border-radius: 4px; padding: 6px;")
        
        self.CaplineEdit = QLineEdit()
        self.CaplineEdit.setText("未开启")
        self.CaplineEdit.setReadOnly(True)
        self.CaplineEdit.setFixedHeight(36)
        self.CaplineEdit.setStyleSheet("border: 1px solid #dcdfe6; border-radius: 4px; padding: 6px; color: #909399;")
        
        layout.addRow("图片文件", self.PiclineEdit)
        layout.addRow("视频文件", self.VideolineEdit)
        layout.addRow("摄像头状态", self.CaplineEdit)
        
        parent_layout.addWidget(self.source_group)

    def create_table_group(self, parent_layout):
        """表格区域 - 保持原样"""
        self.groupBox_3 = QGroupBox("检测结果与位置信息")
        self.groupBox_3.setFont(self.group_font)
        self.groupBox_3.setMinimumHeight(250)
        self.groupBox_3.setMaximumHeight(300)
        self.groupBox_3.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout(self.groupBox_3)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(["序号", "文件路径", "类别", "置信度", "坐标位置"])
        self.tableWidget.setFont(self.table_font)
        
        header_font = QFont("Microsoft YaHei", 11, QFont.Bold)
        self.tableWidget.horizontalHeader().setFont(header_font)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.verticalHeader().setDefaultSectionSize(36)
        
        layout.addWidget(self.tableWidget)
        parent_layout.addWidget(self.groupBox_3)

    def connect_signals(self):
        self.PicBtn.clicked.connect(self.open_img)
        self.FilesBtn.clicked.connect(self.detact_batch_imgs)
        self.VideoBtn.clicked.connect(self.vedio_show)
        self.CapBtn.clicked.connect(self.camera_show)
        self.SaveBtn.clicked.connect(self.save_detect_video)
        self.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.comboBox = QComboBox()
        self.comboBox.activated.connect(self.combox_change)
        
        self.doubleSpinBox.valueChanged.connect(self.update_conf_thres)
        self.doubleSpinBox_2.valueChanged.connect(self.update_iou_thres)
        self.checkBox.stateChanged.connect(self.update_show_labels)
        self.timer_camera.timeout.connect(self.open_frame)

    def combox_change(self):
        if not hasattr(self, 'results') or self.results is None:
            return
            
        com_text = self.comboBox.currentText()
        if com_text == '全部':
            cur_img = self.results.plot()
            if self.cls_list:
                self.current_type_label.setText(f"类型：{Config.CH_names[self.cls_list[0]]}")
                self.current_conf_label.setText(f"置信度：{self.conf_list[0]}")
                xmin, ymin, xmax, ymax = self.location_list[0]
                self.current_pos_label.setText(f"位置：[{xmin}, {ymin}, {xmax}, {ymax}]")
        else:
            try:
                index = int(com_text.split('_')[-1])
                if index < len(self.results):
                    cur_img = self.results[index].plot()
                    self.current_type_label.setText(f"类型：{Config.CH_names[self.cls_list[index]]}")
                    self.current_conf_label.setText(f"置信度：{self.conf_list[index]}")
                    xmin, ymin, xmax, ymax = self.location_list[index]
                    self.current_pos_label.setText(f"位置：[{xmin}, {ymin}, {xmax}, {ymax}]")
                else:
                    return
            except:
                return
        
        self.display_image(cur_img)

    def open_img(self):
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.CaplineEdit.setText('未开启')
            self.cap = None

        file_path, _ = QFileDialog.getOpenFileName(
            None, '打开图片', './', 
            "Image files (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file_path:
            return

        self.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        t1 = time.time()
        self.results = self.model(self.org_path, conf=self.conf_thres, iou=self.iou_thres)[0]
        t2 = time.time()
        detect_time = t2 - t1
        self.metric_time.setText(f'{detect_time:.2f}s')

        self.parse_results()
        
        # 更新指标卡
        if self.conf_list:
            avg_conf = sum([float(c.strip('%')) for c in self.conf_list]) / len(self.conf_list)
            self.metric_conf.setText(f'{avg_conf:.1f}%')
        else:
            self.metric_conf.setText('0%')
        self.metric_targets.setText(str(len(self.cls_list)))
        
        # 更新当前选中目标信息
        if len(self.cls_list) > 0:
            self.current_type_label.setText(f"类型：{Config.CH_names[self.cls_list[0]]}")
            self.current_conf_label.setText(f"置信度：{self.conf_list[0]}")
            xmin, ymin, xmax, ymax = self.location_list[0]
            self.current_pos_label.setText(f"位置：[{xmin}, {ymin}, {xmax}, {ymax}]")
        
        self.display_image(self.results.plot())
        self.PiclineEdit.setText(file_path)
        self.update_ui_from_results()

    def parse_results(self):
        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]

    def display_image(self, img):
        self.draw_img = img
        h, w = img.shape[:2]
        label_w = self.label_show.width()
        label_h = self.label_show.height()
        
        if label_w > 0 and label_h > 0:
            scale = min(label_w / w, label_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(img, (new_w, new_h))
                pix_img = tools.cvimg_to_qpiximg(resized)
                self.label_show.setPixmap(pix_img)
        else:
            pix_img = tools.cvimg_to_qpiximg(img)
            self.label_show.setPixmap(pix_img)

    def update_ui_from_results(self):
        target_nums = len(self.cls_list)
        
        # 更新下拉框
        choose_list = ['全部']
        target_names = [Config.names[id] + '_' + str(index) 
                        for index, id in enumerate(self.cls_list)]
        choose_list.extend(target_names)
        
        self.comboBox.clear()
        self.comboBox.addItems(choose_list)

        self.tableWidget.setRowCount(0)
        self.tabel_info_show(self.location_list, self.cls_list, 
                             self.conf_list, path=self.org_path)

    def tabel_info_show(self, locations, clses, confs, path=None):
        for i, (location, cls, conf) in enumerate(zip(locations, clses, confs)):
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            
            item_id = QTableWidgetItem(str(row+1))
            item_id.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row, 0, item_id)
            
            short_path = path if len(str(path)) <= 40 else '...' + str(path)[-37:]
            self.tableWidget.setItem(row, 1, QTableWidgetItem(short_path))
            
            item_cls = QTableWidgetItem(Config.CH_names[cls])
            item_cls.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row, 2, item_cls)
            
            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row, 3, item_conf)
            
            self.tableWidget.setItem(row, 4, QTableWidgetItem(str(location)))

    def detact_batch_imgs(self):
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.CaplineEdit.setText('未开启')
            self.cap = None
            
        directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        if not directory:
            return
            
        self.org_path = directory
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if (os.path.isfile(full_path) and 
                file_name.split('.')[-1].lower() in img_suffix):
                
                self.org_img = tools.img_cvread(full_path)
                
                t1 = time.time()
                self.results = self.model(full_path, conf=self.conf_thres, 
                                          iou=self.iou_thres)[0]
                t2 = time.time()
                detect_time = t2 - t1
                self.metric_time.setText(f'{detect_time:.2f}s')
                
                self.parse_results()
                if self.conf_list:
                    avg_conf = sum([float(c.strip('%')) for c in self.conf_list]) / len(self.conf_list)
                    self.metric_conf.setText(f'{avg_conf:.1f}%')
                self.metric_targets.setText(str(len(self.cls_list)))
                
                self.display_image(self.results.plot())
                self.PiclineEdit.setText(full_path)
                self.update_ui_from_results()
                self.tableWidget.scrollToBottom()
                QApplication.processEvents()

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None, '打开视频', './', 
            "Video files (*.avi *.mp4 *.wmv *.mkv)"
        )
        if file_path:
            self.org_path = file_path
            self.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        self.tableWidget.setRowCount(0)
        self.comboBox.clear()
        self.timer_camera.start(30)

    def video_stop(self):
        if self.cap:
            self.cap.release()
        self.timer_camera.stop()

    def open_frame(self):
        if not self.cap:
            return
            
        ret, now_img = self.cap.read()
        if ret:
            t1 = time.time()
            results = self.model(now_img, conf=self.conf_thres, iou=self.iou_thres)[0]
            t2 = time.time()
            detect_time = t2 - t1
            self.metric_time.setText(f'{detect_time:.2f}s')
            
            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]
            
            if self.conf_list:
                avg_conf = sum([float(c.strip('%')) for c in self.conf_list]) / len(self.conf_list)
                self.metric_conf.setText(f'{avg_conf:.1f}%')
            self.metric_targets.setText(str(len(self.cls_list)))
            
            now_img = results.plot()
            self.display_image(now_img)
            
            choose_list = ['全部']
            target_names = [Config.names[id] + '_' + str(index) 
                            for index, id in enumerate(self.cls_list)]
            choose_list.extend(target_names)
            self.comboBox.clear()
            self.comboBox.addItems(choose_list)
            
            self.tabel_info_show(self.location_list, self.cls_list, 
                                 self.conf_list, path=self.org_path)
        else:
            self.video_stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.CaplineEdit.setText('未开启')
            
        video_path = self.get_video_path()
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.video_start()
            self.comboBox.setDisabled(True)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.CaplineEdit.setText('已开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.comboBox.setDisabled(True)
        else:
            self.CaplineEdit.setText('未开启')
            if self.cap:
                self.cap.release()
            self.label_show.clear()

    def save_detect_video(self):
        if not self.cap and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap and self.org_path and os.path.isfile(self.org_path):
            res = QMessageBox.information(
                self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if res == QMessageBox.Yes:
                self.video_stop()
                self.btn2Thread_object = btn2Thread(
                    self.org_path, self.model, self.conf_thres, self.iou_thres
                )
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
        elif self.org_path and os.path.isfile(self.org_path):
            fileName = os.path.basename(self.org_path)
            name, end_name = fileName.rsplit(".", 1)
            save_name = name + '_detect_result.' + end_name
            save_img_path = os.path.join(Config.save_path, save_name)
            cv2.imwrite(save_img_path, self.draw_img)
            QMessageBox.about(self, '提示', f'图片保存成功!\n文件路径:{save_img_path}')

    def update_process_bar(self, cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', f'视频保存成功!\n文件在{Config.save_path}目录下')
            return
        if not self.progress_bar.isVisible():
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total * 100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    def update_conf_thres(self, value):
        self.conf_thres = value
        if hasattr(self, 'model') and self.model:
            self.model.conf = value
            if hasattr(self, 'org_img') and self.org_img is not None:
                self.detect_current_image()

    def update_iou_thres(self, value):
        self.iou_thres = value
        if hasattr(self, 'model') and self.model:
            self.model.iou = value
            if hasattr(self, 'org_img') and self.org_img is not None:
                self.detect_current_image()

    def update_show_labels(self, state):
        self.show_labels = state == Qt.Checked
        if hasattr(self, 'results') and self.results is not None:
            self.draw_detection_results()

    def detect_current_image(self):
        if not hasattr(self, 'org_img') or self.org_img is None:
            return
            
        t1 = time.time()
        self.results = self.model(self.org_img, conf=self.conf_thres, iou=self.iou_thres)[0]
        t2 = time.time()
        detect_time = t2 - t1
        self.metric_time.setText(f'{detect_time:.2f}s')
        
        self.parse_results()
        if self.conf_list:
            avg_conf = sum([float(c.strip('%')) for c in self.conf_list]) / len(self.conf_list)
            self.metric_conf.setText(f'{avg_conf:.1f}%')
        self.metric_targets.setText(str(len(self.cls_list)))
        
        self.update_ui_from_results()
        self.draw_detection_results()

    def draw_detection_results(self):
        if not hasattr(self, 'results') or self.results is None:
            return
        
        if self.show_labels:
            now_img = self.results.plot()
        else:
            now_img = self.org_img.copy()
            for box in self.results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                color = self.colors(cls, True)
                cv2.rectangle(now_img, (x1, y1), (x2, y2), color, 2)
        
        self.display_image(now_img)


class btn2Thread(QThread):
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path, model, conf, iou):
        super().__init__()
        self.org_path = path
        self.model = model
        self.conf = conf
        self.iou = iou
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.org_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        fileName = os.path.basename(self.org_path)
        name, _ = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cur_num = 0

        while cap.isOpened() and self.is_running:
            cur_num += 1
            ret, frame = cap.read()
            if ret:
                results = self.model(frame, conf=self.conf, iou=self.iou)[0]
                frame = results.plot()
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
                
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())