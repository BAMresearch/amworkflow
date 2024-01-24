##Author github user @Tanneguydv, 2021

import os
import sys
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QGroupBox,
    QDialog,
    QVBoxLayout,
)
from OCC.Display.backend import load_backend
import importlib
load_backend("qt-pyqt5")
import OCC.Display.qtDisplay as qtDisplay

class Worker(QObject):
    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        
    finished = pyqtSignal()
    reload_flow = pyqtSignal()

    def run(self):
        """Reload geom function from script."""
        self.geometry.create_draft()
        self.reload_flow.emit(self.geometry.draft)
        self.finished.emit()

class App(QDialog):
    def __init__(self, geometry, workflow):
        super().__init__()
        self.title = "PyQt5 / pythonOCC"
        self.left = 300
        self.top = 300
        self.width = 1500
        self.height = 800
        self.geometry = geometry
        self.workflow = workflow
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.createVerticalLayout()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.verticalGroupBox)
        self.setLayout(windowLayout)
        self.show()

    def createVerticalLayout(self):
        self.verticalGroupBox = QGroupBox("Display PythonOCC")
        layout = QVBoxLayout()

        disp = QPushButton("Display Geometry", self)
        disp.clicked.connect(self.displayGEOM)
        disp.setGeometry(50, 10, 50, 30)
        layout.addWidget(disp)
        
        eras = QPushButton("Erase Geometry", self)
        eras.clicked.connect(self.eraseGEOM)
        layout.addWidget(eras)

        self.canvas = qtDisplay.qtViewer3d(self)
        self.canvas.resize(1400,700)
        layout.addWidget(self.canvas)
        
        parent_rect = self.rect()
        canvas_rect = self.canvas.rect()

        # Calculate the center position within the parent's dimensions
        center_x = (parent_rect.width() - canvas_rect.width()) // 2
        center_y = (parent_rect.height() - canvas_rect.height()) // 2

        # Set the center position for the canvas
        self.canvas.move(center_x, center_y)
        self.canvas.InitDriver()
        self.display = self.canvas._display
        self.verticalGroupBox.setLayout(layout)

    def displayGEOM(self):
        self.workflow.geometry_spawn = self.geometry
        self.workflow.create_draft()
        self.draft = self.workflow.draft
        self.geom_display = self.display.DisplayShape(self.draft)[0]
        self.display.FitAll()

    def eraseGEOM(self):
        if hasattr(self, "geom_display"):
            self.display.Context.Erase(self.geom_display, True)

def draft_ui(func: callable, workflow: callable):
    app = QApplication(sys.argv)
    gui = App(func,workflow)
    sys.exit(app.exec_())
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     ex = App()
#     sys.exit(app.exec_())