import sys
import os
print(os.path.dirname(os.path.dirname(__file__)))
from src.constants.enums import Directory
from src.utils.writer import stl_writer
from src.interface.main_window import main_window_frame, ExampleQWidget, ExampleMainWindow, ExampleRealWindow
from PyQt5.QtWidgets import  QApplication
t = 3
match t:
    case 0:    
        main_window_frame()
    case 1:     
        app = QApplication(sys.argv)
        ex = ExampleQWidget()
        sys.exit(app.exec())
    case 2:
        app = QApplication(sys.argv)
        ex = ExampleMainWindow()
        sys.exit(app.exec())
    case 3:
        app = QApplication(sys.argv)
        ex = ExampleRealWindow()
        #sys.stdin.read()
        sys.exit(app.exec())
        

            
