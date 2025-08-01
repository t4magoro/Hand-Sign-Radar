# General Library Imports
import json
import time
from serial.tools import list_ports
import os
import sys
from contextlib import suppress
import matplotlib.cm as cm
import cv2  # pastikan opencv-python sudah diinstall
# PyQt Imports     
from PySide2 import QtGui
from OpenGL.GL import glColor4f
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QKeySequence, QImage, QPixmap        
from PySide2.QtWidgets import (
    QAction,
    QTabWidget,
    QGridLayout,
    QMenu,
    QGroupBox,
    QLineEdit,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QWidget,
    QShortcut,
    QSlider,
    QCheckBox,
    QVBoxLayout,
    QSpacerItem,
    QSizePolicy
)

# Local Imports
from cached_data import CachedDataType
from demo_defines import *
from common.gui_threads import *
from parseFrame import parseStandardFrame

from Common_Tabs.plot_1d import Plot1D
from Common_Tabs.plot_2d import Plot2D
from Common_Tabs.plot_3d import Plot3D

from Demo_Classes.surface_classification import SurfaceClassification
from Demo_Classes.people_tracking import PeopleTracking
from Demo_Classes.gesture_recognition import GestureRecognition
from Demo_Classes.level_sensing import LevelSensing
from Demo_Classes.small_obstacle import SmallObstacle
from Demo_Classes.out_of_box_x843 import OOBx843
from Demo_Classes.out_of_box_x432 import OOBx432
from Demo_Classes.true_ground_speed import TrueGroundSpeed
from Demo_Classes.long_range_pd import LongRangePD
from Demo_Classes.mobile_tracker import MobileTracker
from Demo_Classes.kick_to_open import KickToOpen
from Demo_Classes.calibration import Calibration
from Demo_Classes.vital_signs import VitalSigns
from Demo_Classes.dashcam import Dashcam
from Demo_Classes.ebikes_x432 import EBikes
from Demo_Classes.video_doorbell import VideoDoorbell
from Demo_Classes.two_pass_video_doorbell import TwoPassVideoDoorbell

# Logger
import logging
log = logging.getLogger(__name__)


class Window(QMainWindow):
    def __init__(self, parent=None, size=[], title="Applications Visualizer"):
        super(Window, self).__init__(parent)
        self.core = Core()
        self.heatmap_target_size = (320, 160)
        self.setWindowIcon(QtGui.QIcon("./images/logo.png"))

        self.shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        self.shortcut.activated.connect(self.close)
        self.demoTabs = QTabWidget()
        self.gridLayout = QGridLayout()

        self.initConfigPane()
        self.initConnectionPane()

        self.gridLayout.addWidget(self.comBox, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.configBox, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.demoTabs, 0, 1, 8, 1)

        # 🔹 Voxel Panel - harus dideklarasikan dulu sebelum ditambahkan
        self.voxelLayout = QVBoxLayout()
        self.voxelLayout.setContentsMargins(10, 10, 10, 10)
        self.voxelLayout.setSpacing(8)

        # voxelTitle = QLabel("Visualisasi 3D Voxel")
        # voxelTitle.setAlignment(Qt.AlignCenter)
        # voxelTitle.setStyleSheet("font-weight: bold; font-size: 14px;")

        # self.voxelLabel = QLabel()
        # self.voxelLabel.setAlignment(Qt.AlignCenter)
        # self.voxelLabel.setStyleSheet("background-color: black; border: none;")
        # self.voxelLabel.setFixedSize(384, 192)

        # self.voxelLayout.addWidget(voxelTitle)
        # self.voxelLayout.addWidget(self.voxelLabel)

        # 🔹 Heatmap Panel (DR, DT, RT)
        self.heatmapLayout = QVBoxLayout()
        self.heatmapLayout.setContentsMargins(10, 10, 10, 10)
        self.heatmapLayout.setSpacing(8)

        def create_heatmap_block(title_text, attr_name):
            title = QLabel(title_text)
            title.setAlignment(Qt.AlignCenter)
            title.setFixedWidth(self.heatmap_target_size[0])
            title.setStyleSheet("""
                font-weight: bold;
                font-size: 14px;
                padding-bottom: 4px;
            """)

            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: black; border: none;")
            label.setFixedSize(*self.heatmap_target_size)

            setattr(self, attr_name, label)

            self.heatmapLayout.addWidget(title)
            self.heatmapLayout.addWidget(label)

        # Tambahkan heatmaps DR, DT, RT
        for title, var in [
            ("Doppler vs x", "labelDR"),
            ("Doppler vs y", "labelDT"),
            ("Doppler vs Z", "labelRT"),
        ]:
            create_heatmap_block(title, var)

        # # ⬇️ Tambahkan voxel panel di bawah heatmap
        # self.heatmapLayout.addSpacing(12)
        # self.heatmapLayout.addLayout(self.voxelLayout)

        # Spacer bawah agar tidak mepet
        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.heatmapLayout.addItem(spacer)
        # 🔤 Label Prediksi Aktivitas
        self.predictionLabel = QLabel("Aktivitas: -")
        self.predictionLabel.setAlignment(Qt.AlignCenter)
        self.predictionLabel.setStyleSheet("""
            font-weight: bold;
            font-size: 60px;
            color: green;
        """)
        self.heatmapLayout.addWidget(self.predictionLabel)


        # Bungkus ke widget
        self.heatmapContainer = QWidget()
        self.heatmapContainer.setLayout(self.heatmapLayout)
        self.gridLayout.addWidget(self.heatmapContainer, 0, 2, 8, 1)

        # Atur grid kolom
        self.gridLayout.setColumnStretch(0, 1)  # kiri (COM/config)
        self.gridLayout.setColumnStretch(1, 5)  # tengah (tabs/3D)
        self.gridLayout.setColumnStretch(2, 3)  # kanan (heatmap+voxel)

        # Replay slider (bawaan)
        self.core.sl.setMinimum(0)
        self.core.sl.setMaximum(30)
        self.core.sl.setValue(20)
        self.core.sl.setTickPosition(QSlider.TicksBelow)
        self.core.sl.setTickInterval(5)

        self.replayBox = QGroupBox("Replay")
        self.replayLayout = QGridLayout()
        self.replayLayout.addWidget(self.core.sl, 0, 0, 1, 1)
        self.replayBox.setLayout(self.replayLayout)
        self.replayBox.setVisible(False)
        self.gridLayout.addWidget(self.replayBox, 8, 0, 1, 2)

        # Set ke central layout
        self.central = QWidget()
        self.central.setLayout(self.gridLayout)
        self.setCentralWidget(self.central)

        self.setWindowTitle(title)
        self.initMenuBar()
        self.core.replay = False

        self.showMaximized()

        self.core.sl.setMinimum(0)
        self.core.sl.setMaximum(30)
        self.core.sl.setValue(20)
        self.core.sl.setTickPosition(QSlider.TicksBelow)
        self.core.sl.setTickInterval(5)

        self.replayBox = QGroupBox("Replay")
        self.replayLayout = QGridLayout()
        self.replayLayout.addWidget(self.core.sl, 0, 0, 1, 1)
        self.replayBox.setLayout(self.replayLayout)
        self.replayBox.setVisible(False)
        self.gridLayout.addWidget(self.replayBox, 8, 0, 1, 2)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 5)

        self.central = QWidget()
        self.central.setLayout(self.gridLayout)

        self.setWindowTitle("Industrial Visualizer")  # ✅ Benar!
        self.initMenuBar()
        self.core.replay = False

        self.setCentralWidget(self.central)

        self.showMaximized()

    def initMenuBar(self):
        menuBar = self.menuBar()
        # Creating menus using a QMenu object
        fileMenu = QMenu("&File", self)
        playbackMenu = QMenu("&Playback", self)

        self.logOutputAction = QAction("Log Terminal Output to File", self)
        self.playbackAction = QAction("Load and Replay", self)

        self.playbackAction.triggered.connect(self.loadForReplay)
        self.playbackAction.setCheckable(True)
        self.logOutputAction.triggered.connect(self.toggleLogOutput)
        self.logOutputAction.setCheckable(True)

        playbackMenu.addAction(self.playbackAction)
        fileMenu.addAction(self.logOutputAction)
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(playbackMenu)

    def loadForReplay(self, state):
        if (state):
            self.recordAction.setChecked(False)
            self.core.replayFile = QFileDialog.getOpenFileName(self, 'Open Replay JSON File', '.',"JSON Files (*.json)")
            self.core.replay = True
            self.core.loadForReplay(True)

            # Disable COM Ports/Device/Demo/Config
            self.demoList.setEnabled(False)
            self.deviceList.setEnabled(False)
            self.cliCom.setEnabled(False)
            self.dataCom.setEnabled(False)
            self.connectButton.setEnabled(False)
            self.filename_edit.setEnabled(False)
            self.selectConfig.setEnabled(False)
            self.sendConfig.setEnabled(False)
            self.start.setEnabled(True)
            self.start.setText("Replay")

            self.replayBox.setVisible(True)
        else:
            self.core.replay = False

            # Disable COM Ports/Device/Demo/Config
            self.demoList.setEnabled(True)
            self.deviceList.setEnabled(True)
            self.cliCom.setEnabled(True)
            self.dataCom.setEnabled(True)
            self.connectButton.setEnabled(True)
            self.filename_edit.setEnabled(True)
            self.selectConfig.setEnabled(True)
            self.sendConfig.setEnabled(True)
            self.start.setText("Start without Send Configuration")

            self.replayBox.setVisible(False)

    def toggleSaveData(self):
        if self.recordAction.isChecked():
            self.core.parser.setSaveBinary(True)
        else:
            self.core.parser.setSaveBinary(False)
        
        self.core.replay = False
        
        # Enable COM Ports/Device/Demo/Config
        self.demoList.setEnabled(True)
        self.deviceList.setEnabled(True)
        self.cliCom.setEnabled(True)
        self.dataCom.setEnabled(True)
        self.connectButton.setEnabled(True)
        self.filename_edit.setEnabled(True)
        self.selectConfig.setEnabled(True)
        self.start.setText("Start without Send Configuration")

    def toggleLogOutput(self):
        if (
            self.recordAction.isChecked()
        ):  # Save terminal output to logFile, set 0 to show terminal output
            ts = time.localtime()
            terminalFileName = str(
                "logfile_"
                + str(ts[2])
                + str(ts[1])
                + str(ts[0])
                + "_"
                + str(ts[3])
                + str(ts[4])
                + ".txt"
            )
            sys.stdout = open(terminalFileName, "w")
        else:
            sys.stdout = sys.__stdout__

    def initConnectionPane(self):
        self.comBox = QGroupBox("Connect to COM Ports")
        self.cliCom = QLineEdit("")
        self.dataCom = QLineEdit("")
        self.connectStatus = QLabel("Not Connected")
        self.connectButton = QPushButton("Connect")
        self.connectButton.clicked.connect(self.onConnect)
        self.demoList = QComboBox()
        self.deviceList = QComboBox()
        self.recordAction = QCheckBox("Save Data to File", self)

        # TODO Add replay support
        self.demoList.addItems(self.core.getDemoList())
        self.demoList.currentIndexChanged.connect(self.onChangeDemo)
        self.deviceList.addItems(self.core.getDeviceList())
        self.deviceList.currentIndexChanged.connect(self.onChangeDevice)
        self.comLayout = QGridLayout()
        self.comLayout.addWidget(QLabel("Device:"), 0, 0)
        self.comLayout.addWidget(self.deviceList, 0, 1)
        self.comLayout.addWidget(QLabel("CLI COM:"), 1, 0)
        self.comLayout.addWidget(self.cliCom, 1, 1)
        self.comLayout.addWidget(QLabel("DATA COM:"), 2, 0)
        self.comLayout.addWidget(self.dataCom, 2, 1)
        self.comLayout.addWidget(QLabel("Demo:"), 3, 0)
        self.comLayout.addWidget(self.demoList, 3, 1)
        self.comLayout.addWidget(self.connectButton, 4, 0)
        self.recordAction.stateChanged.connect(self.toggleSaveData)
        self.comLayout.addWidget(self.recordAction, 5, 0)
        self.comLayout.addWidget(self.connectStatus, 4, 1)

        self.comBox.setLayout(self.comLayout)
        self.demoList.setCurrentIndex(1)  # initialize this to a stable value
        self.demoList.setCurrentIndex(0)  # initialize this to a stable value

        # Find all Com Ports
        serialPorts = list(list_ports.comports())

        # Find default CLI Port and Data Port
        for port in serialPorts:
            if (
                CLI_XDS_SERIAL_PORT_NAME in port.description
                or CLI_SIL_SERIAL_PORT_NAME in port.description
            ):
                log.info(f"CLI COM Port found: {port.device}")
                comText = port.device
                comText = comText.replace("COM", "")
                self.cliCom.setText(comText)

            elif (
                DATA_XDS_SERIAL_PORT_NAME in port.description
                or DATA_SIL_SERIAL_PORT_NAME in port.description
            ):
                log.info(f"Data COM Port found: {port.device}")
                comText = port.device
                comText = comText.replace("COM", "")
                self.dataCom.setText(comText)

        self.core.isGUILaunched = 1
        self.loadCachedData()

    def initConfigPane(self):
        self.configBox = QGroupBox("Configuration")
        self.selectConfig = QPushButton("Select Configuration")
        self.sendConfig = QPushButton("Start and Send Configuration")
        self.start = QPushButton("Start without Send Configuration")
        self.sensorStop = QPushButton("Send sensorStop Command")
        self.sensorStop.setToolTip("Stop sensor (only works if lowPowerCfg is 0)")
        self.filename_edit = QLineEdit()
        self.selectConfig.clicked.connect(lambda: self.selectCfg(self.filename_edit))
        self.sendConfig.setEnabled(False)
        self.start.setEnabled(False)
        self.sendConfig.clicked.connect(self.sendCfg)
        self.start.clicked.connect(self.startApp)
        self.sensorStop.clicked.connect(self.stopSensor)
        self.sensorStop.setHidden(True)
        self.configLayout = QGridLayout()
        self.configLayout.addWidget(self.filename_edit, 0, 0, 1, 1)
        self.configLayout.addWidget(self.selectConfig, 0, 1, 1, 1)
        self.configLayout.addWidget(self.sendConfig, 1, 0, 1, 2)
        self.configLayout.addWidget(self.start, 2, 0, 1, 2)
        self.configLayout.addWidget(self.sensorStop, 3, 0, 1, 2)
        # self.configLayout.addStretch(1)
        self.configBox.setLayout(self.configLayout)

    def loadCachedData(self):
        self.core.loadCachedData(
            self.demoList, self.deviceList, self.recordAction, self.gridLayout, self.demoTabs
        )

    # Callback function when device is changed
    def onChangeDevice(self):
        self.core.changeDevice(
            self.demoList, self.deviceList, self.gridLayout, self.demoTabs
        )
        self.core.updateCOMPorts(self.cliCom, self.dataCom)
        self.core.updateResetButton(self.sensorStop)

    # Callback function when demo is changed
    def onChangeDemo(self):
        self.core.changeDemo(
            self.demoList, self.deviceList, self.gridLayout, self.demoTabs
        )
        # When 2-Pass Video doorbell is the demo, you cannot send a cfg file over UART
        if(self.core.demo == "2-Pass Video Doorbell"):
            self.sendConfig.setDisabled(1)
        else:
            self.sendConfig.setDisabled(0)

        # self.core.changeDevice(self.demoList, self.deviceList, self.gridLayout, self.demoTabs)

    # Callback function when connect button clicked
    def onConnect(self):
        self.core.parentWindow = self  # ✅ Set dulu
        if (self.connectStatus.text() == "Not Connected" or self.connectStatus.text() == "Unable to Connect"):
            if self.core.connectCom(self.cliCom, self.dataCom, self.connectStatus) == 0:
                self.connectButton.setText("Reset Connection")
                self.sendConfig.setEnabled(True)
                self.start.setEnabled(True)
            else:
                self.sendConfig.setEnabled(False)
                self.start.setEnabled(False)
        else:
            self.core.gracefulReset()
            self.connectButton.setText("Connect")
            self.connectStatus.setText("Not Connected")
            self.sendConfig.setEnabled(False)
            self.start.setEnabled(False)

            # need to do ser.close()

    # Callback function when 'Select Configuration' is clicked
    def selectCfg(self, filename):
        self.core.selectCfg(filename)

    # Callback function when 'Start and Send Configuration' is clicked
    def sendCfg(self):
        self.core.sendCfg()

    # Callback function to send sensorStop to device
    def stopSensor(self):
        self.core.stopSensor()

    # Callback function when 'Start without Send Configuration' is clicked
    def startApp(self):
        if (self.core.replay and self.core.playing is False):
            self.start.setText("Pause")
        elif (self.core.replay and self.core.playing is True):
            self.start.setText("Replay")
        self.core.startApp()

    def updateHeatmapGUI(self, dr, dt, rt):
        target_size = self.heatmap_target_size
        dr = cv2.resize(dr, target_size, interpolation=cv2.INTER_CUBIC)
        dt = cv2.resize(dt, target_size, interpolation=cv2.INTER_CUBIC)
        rt = cv2.resize(rt, target_size, interpolation=cv2.INTER_CUBIC)

        self.labelDR.setPixmap(self.heatmap_to_pixmap(dr))
        self.labelDT.setPixmap(self.heatmap_to_pixmap(dt))
        self.labelRT.setPixmap(self.heatmap_to_pixmap(rt))

    def heatmap_to_pixmap(self, heatmap):
        # Jika hanya 2 dimensi (grayscale), ubah jadi RGB pakai colormap
        if len(heatmap.shape) == 2:
            cmap = cm.get_cmap("jet")  # atau "viridis", bebas
            heatmap_colored = cmap(heatmap)[:, :, :3]  # Ambil R,G,B saja
            heatmap_rgb = (heatmap_colored * 255).astype(np.uint8)
        else:
            heatmap_rgb = (heatmap * 255).astype(np.uint8)

        h, w, ch = heatmap_rgb.shape
        bytesPerLine = ch * w
        qimg = QImage(heatmap_rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


class Core:
    def __init__(self):
        self.cachedData = CachedDataType()

        self.device = "xWR6843"
        self.demo = DEMO_OOB_x843

        self.frameTime = 50
        self.parser = UARTParser(type="DoubleCOMPort")

        self.replayFile = "replay.json"
        self.replay = False

        # set to 1 
        self.isGUILaunched = 0

        self.sl = QSlider(Qt.Horizontal)
        self.sl.valueChanged.connect(self.sliderValueChange)
        self.playing = False
        self.replayFrameNum = 0

        # Populated with each demo and it's corresponding object
        self.demoClassDict = {
            DEMO_OOB_x843: OOBx843(),
            DEMO_OOB_x432: OOBx432(),
            DEMO_3D_PEOPLE_TRACKING: PeopleTracking(),
            DEMO_VITALS: VitalSigns(),
            DEMO_SMALL_OBSTACLE: SmallObstacle(),
            DEMO_GESTURE: GestureRecognition(),
            DEMO_SURFACE: SurfaceClassification(),
            DEMO_LEVEL_SENSING: LevelSensing(),
            DEMO_GROUND_SPEED: TrueGroundSpeed(),
            DEMO_LONG_RANGE: LongRangePD(),
            DEMO_MOBILE_TRACKER: MobileTracker(),
            DEMO_KTO: KickToOpen(),
            DEMO_CALIBRATION: Calibration(),
            DEMO_DASHCAM: Dashcam(),
            DEMO_EBIKES: EBikes(),
            DEMO_VIDEO_DOORBELL: VideoDoorbell(),
            DEMO_TWO_PASS_VIDEO_DOORBELL: TwoPassVideoDoorbell(),
        }

    def loadCachedData(self, demoList, deviceList, recordAction, gridLayout, demoTabs):
        deviceName = self.cachedData.getCachedDeviceName()
        demoName = self.cachedData.getCachedDemoName()
        recordState = bool(self.cachedData.getCachedRecord())

        if deviceName in self.getDeviceList():
            deviceList.setCurrentIndex(self.getDeviceList().index(deviceName))

        if demoName in self.getDemoList():
            demoList.setCurrentIndex(self.getDemoList().index(demoName))
            self.changeDemo(demoList, deviceList, gridLayout, demoTabs)

        if recordState:
            recordAction.setChecked(False)

    def getDemoList(self):
        return DEVICE_DEMO_DICT[self.device]["demos"]

    def getDeviceList(self):
        return list(DEVICE_DEMO_DICT.keys())

    def changeDemo(self, demoList, deviceList, gridLayout, demoTabs):
        self.demo = demoList.currentText()

        if (self.isGUILaunched):
            self.cachedData.setCachedDemoName(self.demo)
            self.cachedData.setCachedDeviceName(deviceList.currentText())

        permanentWidgetsList = ["Connect to COM Ports", "Configuration", "Tabs", "Replay"]
        # Destroy current contents of graph pane
        for _ in range(demoTabs.count()):
            demoTabs.removeTab(0)
        for i in range(gridLayout.count()):
            try:
                currWidget = gridLayout.itemAt(i).widget()
                if currWidget.title() not in permanentWidgetsList:
                    currWidget.setVisible(False)
            except AttributeError as e:
                log.log(0, "Demo Tabs don't have title attribute. This is OK")
                continue

        # Make call to selected demo's initialization function
        if self.demo in self.demoClassDict:
            self.demoClassDict[self.demo].setupGUI(gridLayout, demoTabs, self.device)

    def changeDevice(self, demoList, deviceList, gridLayout, demoTabs):
        self.device = deviceList.currentText()

        if (self.isGUILaunched):
            self.cachedData.setCachedDemoName(demoList.currentText())
            self.cachedData.setCachedDeviceName(self.device)

        if DEVICE_DEMO_DICT[self.device]["singleCOM"]:
            self.parser.parserType = "SingleCOMPort"
        else:
            self.parser.parserType = "DoubleCOMPort"

        demoList.clear()
        demoList.addItems(DEVICE_DEMO_DICT[self.device]["demos"])

    def updateCOMPorts(self, cliCom, dataCom):
        if DEVICE_DEMO_DICT[self.device]["isxWRLx432"]:
            dataCom.setText(cliCom.text())
            dataCom.setEnabled(False)
        else:
            dataCom.setEnabled(True)

    def updateResetButton(self, sensorStopButton):
        if DEVICE_DEMO_DICT[self.device]["isxWRLx432"]:
            sensorStopButton.setHidden(True) # TODO change to false once sending sensorStop is implemented
        else:
            sensorStopButton.setHidden(True)

    def stopSensor(self):
        self.parser.sendLine("sensorStop 0")

    def selectFile(self, filename):
        try:
            current_dir = os.getcwd()
            configDirectory = current_dir
            path = self.cachedData.getCachedCfgPath()
            if path != "":
                configDirectory = path
        except:
            configDirectory = ""

        fname = QFileDialog.getOpenFileName(caption="Open .cfg File", dir=configDirectory, filter="cfg(*.cfg)")
        filename.setText(str(fname[0]))
        return fname[0]

    def parseCfg(self, fname):
        if (self.replay):
            self.cfg = self.data['cfg']
        else:
            with open(fname, "r") as cfg_file:
                self.cfg = cfg_file.readlines()
                self.parser.cfg = self.cfg
                self.parser.demo = self.demo
                self.parser.device = self.device
        for line in self.cfg:
            args = line.split()
            print(args)
            if len(args) > 0:
                # trackingCfg
                if args[0] == "trackingCfg":
                    if len(args) < 5:
                        log.error("trackingCfg had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseTrackingCfg(args)
                elif args[0] == "SceneryParam" or args[0] == "boundaryBox":
                    if len(args) < 7:
                        log.error(
                            "SceneryParam/boundaryBox had fewer arguments than expected"
                        )
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseBoundaryBox(args)
                elif args[0] == "frameCfg":
                    if len(args) < 4:
                        log.error("frameCfg had fewer arguments than expected")
                    else:
                        self.frameTime = float(args[5]) / 2
                elif args[0] == "sensorPosition":
                    # sensorPosition for x843 family has 3 args
                    if DEVICE_DEMO_DICT[self.device]["isxWRx843"] and len(args) < 4:
                        log.error("sensorPosition had fewer arguments than expected")
                    elif DEVICE_DEMO_DICT[self.device]["isxWRLx432"] and len(args) < 6:
                        log.error("sensorPosition had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseSensorPosition(
                                args, DEVICE_DEMO_DICT[self.device]["isxWRx843"]
                            )
                # Only used for Small Obstacle Detection
                elif args[0] == "occStateMach":
                    numZones = int(args[1])
                # Only used for Small Obstacle Detection
                elif args[0] == "zoneDef":
                    if len(args) < 8:
                        log.error("zoneDef had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseBoundaryBox(args)
                elif args[0] == "mpdBoundaryBox":
                    if len(args) < 8:
                        log.error("mpdBoundaryBox had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseBoundaryBox(args)
                elif args[0] == "chirpComnCfg":
                    if len(args) < 8:
                        log.error("chirpComnCfg had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseChirpComnCfg(args)
                elif args[0] == "chirpTimingCfg":
                    if len(args) < 6:
                        log.error("chirpTimingCfg had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseChirpTimingCfg(args)
                # TODO This is specifically guiMonitor for 60Lo, this parsing will break the gui when an SDK 3 config is sent
                elif args[0] == "guiMonitor":
                    if DEVICE_DEMO_DICT[self.device]["isxWRLx432"]:
                        if len(args) < 12:
                            log.error("guiMonitor had fewer arguments than expected")
                        else:
                            with suppress(AttributeError):
                                self.demoClassDict[self.demo].parseGuiMonitor(args)
                elif args[0] == "presenceDetectCfg":
                    with suppress(AttributeError):
                        self.demoClassDict[self.demo].parsePresenceDetectCfg(args)
                elif args[0] == "sigProcChainCfg2":
                    with suppress(AttributeError):
                        self.demoClassDict[self.demo].parseSigProcChainCfg2(args)
                elif args[0] == "mpdBoundaryArc":
                    if len(args) < 8:
                        log.error("mpdBoundaryArc had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseBoundaryBox(args)
                elif args[0] == "measureRangeBiasAndRxChanPhase":
                    with suppress(AttributeError):
                        self.demoClassDict[self.demo].parseRangePhaseCfg(args)
                elif args[0] == "clutterRemoval":
                    with suppress(AttributeError):
                        self.demoClassDict[self.demo].parseClutterRemovalCfg(args)
                elif args[0] == "sigProcChainCfg":
                    with suppress(AttributeError):
                        self.demoClassDict[self.demo].parseSigProcChainCfg(args)
                elif args[0] == "channelCfg":
                    with suppress(AttributeError):
                        self.demoClassDict[self.demo].parseChannelCfg(args)

        # Initialize 1D plot values based on cfg file
        with suppress(AttributeError):
            self.demoClassDict[self.demo].setRangeValues()

    def selectCfg(self, filename):
        try:
            file = self.selectFile(filename)
            self.cachedData.setCachedCfgPath(file)  # cache the file and demo used
            self.parseCfg(file)
        except Exception as e:
            log.error(e)
            log.error(
                "Parsing .cfg file failed. Did you select a valid configuration file?"
            )

        log.debug("Demo Changed to " + self.demo)
        if self.demo == DEMO_CALIBRATION:
            self.demoClassDict[self.demo].checkCalibrationParams()

    def sendCfg(self):
        try:
            if self.demo != "Replay":
                self.parser.sendCfg(self.cfg)
                sys.stdout.flush()
                self.parseTimer.start(int(self.frameTime))  # need this line
        except Exception as e:
            log.error(e)
            log.error("Parsing .cfg file failed. Did you select the right file?")

    def updateGraph(self, outputDict):
        self.demoClassDict[self.demo].updateGraph(outputDict)

    def connectCom(self, cliCom, dataCom, connectStatus):
        if self.demo == DEMO_GESTURE:
            self.frameTime = 25 # Gesture demo runs at 35ms frame time
        # init threads and timers
        # self.uart_thread = parseUartThread(self.parser, viewer=self.heatmapViewer)
        self.uart_thread = parseUartThread(self.parser)
        self.uart_thread.guiWindow = self.parentWindow  # ⬅️ ini harus menunjuk ke GUI Window
        self.uart_thread.fin.connect(self.updateGraph)
        self.parseTimer = QTimer()
        self.parseTimer.setSingleShot(False)
        self.parseTimer.timeout.connect(self.parseData)
        try:
            if os.name == "nt":
                uart = "COM" + cliCom.text()
                data = "COM" + dataCom.text()
            else:
                uart = cliCom.text()
                data = dataCom.text()
            if DEVICE_DEMO_DICT[self.device]["isxWRx843"]:  # If using x843 device
                self.parser.connectComPorts(uart, data)
            else:  # If not x843 device then defer to x432 device
                if self.demo == DEMO_GESTURE or self.demo == DEMO_KTO or self.demo == DEMO_TWO_PASS_VIDEO_DOORBELL or self.demo == DEMO_VIDEO_DOORBELL:
                    self.parser.connectComPort(uart, 1250000)
                else:
                    self.parser.connectComPort(uart)
            connectStatus.setText("Connected")
        except Exception as e:
            log.error(e)
            connectStatus.setText("Unable to Connect")
            return -1

        return 0

    def startApp(self):
        if (self.replay and self.playing is False):
            self.replayTimer = QTimer()
            self.replayTimer.setSingleShot(True)
            self.replayTimer.timeout.connect(self.replayData)
            self.playing = True
            self.replayTimer.start(100) # arbitrary value to start plotting
        elif (self.replay and self.playing is True):
            self.playing = False
        else :
            self.parseTimer.start(int(self.frameTime))  # need this line, this is for normal plotting

    def loadForReplay(self, state):
        if (state):
            self.cachedData.setCachedRecord = "True"
            with open(self.replayFile[0], 'r') as fp:
                self.data = json.load(fp)
            self.parseCfg("")
            self.sl.setMinimum(0)
            self.sl.setMaximum(len(self.data['data']) - 1)
            self.sl.setValue(0)
            self.sl.setTickInterval(5)
            # TODO need to load correct demo from file
        else:
            self.cachedData.setCachedRecord = "False"


    def replayData(self):
        if (self.playing) :
            outputDict = self.data['data'][self.replayFrameNum]['frameData']
            self.updateGraph(outputDict)
            self.replayFrameNum += 1
            self.sl.setValue(self.replayFrameNum)
            if (self.replayFrameNum < len(self.data['data'])) :
                self.replayTimer.start(self.data['data'][self.replayFrameNum]['timestamp'] - self.data['data'][self.replayFrameNum-1]['timestamp'])

    def sliderValueChange(self):
        self.replayFrameNum = self.sl.value()

    def parseData(self):
        self.uart_thread.start(priority=QThread.HighestPriority)

    def gracefulReset(self):
        self.parseTimer.stop()
        self.uart_thread.stop()
        if self.parser.cliCom is not None:
            self.parser.cliCom.close()
        if self.parser.dataCom is not None:
            self.parser.dataCom.close()
        for demo in self.demoClassDict.values():
            if hasattr(demo, "plot_3d_thread"):
                demo.plot_3d_thread.stop()
            if hasattr(demo, "plot_3d"):
                demo.removeAllBoundBoxes()