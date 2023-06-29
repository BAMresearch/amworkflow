import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip, QMessageBox, QMainWindow, QAction, QVBoxLayout
from PyQt5.QtGui import QFont, QIcon
from OCC.Display.backend import load_backend
load_backend('qt-pyqt5')
from amworkflow.src.geometries.simple_geometry import create_box, create_prism
from OCC.Display.qtDisplay import qtViewer3d
from OCC.Core.AIS import AIS_Shape, AIS_InteractiveObject

def main_window_frame():
    q_app = QApplication(sys.argv)
    q_widget = QWidget()
    q_widget.resize(700, 800)
    q_widget.move(300, 300)
    q_widget.setWindowTitle('AM-Workflow')
    q_widget.show()
    sys.exit(q_app.exec())

class ExampleQWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.center()
        self.setWindowTitle('Message self.box')
        self.show()
        
    def center(self):
        q_rectangle = self.frameGeometry()
        center_point = self.screen().availableGeometry().center()
        q_rectangle.moveCenter(center_point)
        self.move(q_rectangle.topLeft())

class ExampleMainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.statusBar().showMessage('Ready')
        self.center()
        self.setWindowTitle('Statusbar')
        self.show()
        
        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(QApplication.instance().quit)
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction(exitAct)
        
    def center(self):
        q_rectangle = self.frameGeometry()
        center_point = self.screen().availableGeometry().center()
        q_rectangle.moveCenter(center_point)
        self.move(q_rectangle.topLeft())
    
class ExampleRealWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.central_widget = QWidget(self)
        self.viewer = qtViewer3d(self)
        # self.viewer.InitDriver()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.viewer)
        self.box = create_box(50, 50, 50)
        self.ais_box = AIS_Shape(self.box)
        self.prism = create_prism()
        self.ais_prism = AIS_Shape(self.prism)
        self.viewer._display.Context.Display(self.ais_box, True)
        self.viewer._display.FitAll()
        self.show()