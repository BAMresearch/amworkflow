from enum import Enum
import os.path as op

class Directory(Enum):
    SYS_PATH = op.dirname(op.dirname(op.dirname(op.dirname(__file__))))
    RETURN_ONE_LAYER = "/.."
    FREECAD_PATH = r'/freecad_appimage/squashfs-root/usr/lib/'
    STL_OUTPUT_DIR = ""
    DATABASE_FILE_PATH = op.dirname(op.dirname(__file__)) + r'/infrastructure/database/files/db/'
    DATABASE_OUTPUT_FILE_PATH = op.dirname(op.dirname(__file__)) + r'/infrastructure/database/files/output_files/'
    USECASE_PATH = SYS_PATH + r'/usecases'
    USECASE_PATH_PARAMWALL_PATH = USECASE_PATH + r'/param_wall/'
    TEST_OUTPUT_PATH = SYS_PATH + "/stlOutput/"+"testbatch/"

class Timestamp(Enum):
    YY_MM_DD_HH_MM_SS = "%y%m%d%H%M%S"
    YYYY_MM_DD_HH_MM = "%Y%m%d%H%M"