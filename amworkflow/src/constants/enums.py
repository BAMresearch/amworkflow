from enum import Enum
import os.path as op

class Directory(Enum):
    SYS_PATH = op.dirname(op.dirname(op.dirname(op.dirname(__file__))))
    RETURN_ONE_LAYER = "/.."
    FREECAD_PATH = r'/freecad_appimage/squashfs-root/usr/lib/'
    STL_OUTPUT_DIR = ""