import sys
from OCC.Display.backend import load_backend
load_backend('qt-pyqt5')  # Specify the backend here (e.g., 'qt-pyqt5', 'qt-pyside2', etc.)
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from OCC.Display.qtDisplay import qtViewer3d
from OCC.Core.AIS import AIS_Shape, AIS_InteractiveObject
import pyvista as pv


def td_visualizer(item: AIS_InteractiveObject) -> None :
    app = QApplication(sys.argv)
    window = QMainWindow()
    central_widget = QWidget(window)
    viewer = qtViewer3d()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    layout.addWidget(viewer)
    ais_shape = AIS_Shape(item)
    viewer._display.Context.Display(ais_shape, True)
    viewer._display.FitAll()
    window.show()
    sys.exit(app.exec_())
    
def vtk_visualizer(dirname: str,
                   filename: str):
    vtk_file = dirname + filename
    mesh = pv.read(vtk_file)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, style="wireframe", color="white")
    plotter.show()