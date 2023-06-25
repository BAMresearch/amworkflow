from src.constants.enums import Mapper as M
from src.constants.exceptions import DimensionViolationException
import numpy as np

def import_freecad_check():
    import sys
    from src.constants.enums import Directory
    freecad_path = Directory.SYS_PATH.value + Directory.FREECAD_PATH.value
    sys.path.append(freecad_path)
    try:
        import FreeCAD
        import Part
        return True
    except:
        return False


        
