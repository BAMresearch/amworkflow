# Insert code here
import sys
import pyvista as pv
from pyvista import examples
import pyvistaqt as pvqt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets  # pip3 install pyqt5
from PyQt5.QtCore import *  # QFile, QFileInfo, Qt, QSortFilterProxyModel, QAbstractTableModel
from PyQt5.QtGui import *  # QPen, QStandardItem, QStandardItemModel, QPushButton
from PyQt5.QtWidgets import *  # QStyledItemDelegate, QApplication, QHeaderView, QTableView, QWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, QListWidget, QPushButton, QMenu, QMessageBox,QDialog


class Picker:
    def __init__(self, plotter, mesh):
        self.plotter = plotter
        self.mesh = mesh
        self._points = []

    @property
    def points(self):
        """To access all th points when done."""
        return self._points

    def __call__(self, *args):
        picked_pt = np.array(self.plotter.pick_mouse_position())

        # Line below added to show where mouse position is...
        w.plotter.add_mesh(pv.Sphere(radius=3, center=picked_pt), color='pink')

        direction = picked_pt - self.plotter.camera_position[0]
        direction = direction / np.linalg.norm(direction)
        start = picked_pt - 1000 * direction
        end = picked_pt + 10000 * direction
        point, ix = self.mesh.ray_trace(start, end, first_point=True)
        if len(point) > 0:
            self._points.append(point)
            q = w.plotter.add_mesh(pv.Sphere(radius=3, center=point),
                                   color='red')
        return


class OpNote(QDialog):
    def __init__(self):
        super().__init__()
        mesh = examples.load_airplane()

        self.plotter = pvqt.QtInteractor()
        self.plotter.add_mesh(mesh, show_edges=True, color='w')

        self.tabs.addWidget(self.plotter.interactor)

        picker = Picker(self.plotter, mesh)
        self.plotter.track_click_position(picker, side='right')
        self.plotter.add_text('Use right mouse click to pick points')


app = QtWidgets.QApplication(sys.argv)
w = OpNote()
w.show()
app.exec_()