from amworkflow.src.constants.exceptions import DimensionViolationException, NoDataInDatabaseException, InvalidFileFormatException
import numpy as np
import os

def import_freecad_check():
    import sys
    from amworkflow.src.constants.enums import Directory
    freecad_path = Directory.SYS_PATH.value + Directory.FREECAD_PATH.value
    sys.path.append(freecad_path)
    try:
        import FreeCAD
        import Part
        return True
    except:
        return False

def dimension_check(dm_list: list):
    for dm_item in dm_list:
        if dm_item[3] != 0 or None:
            try:
                assert(dm_item[3] - 0.5 * dm_item[1] > 0)
            except:
                raise DimensionViolationException("Width should be smaller than the Radius")
            try:
                assert(dm_item[0] <= dm_item[3] * 3.1416 * 2)
            except:
                raise DimensionViolationException("Length is too large.")
            
def path_append_check():
    try:
        import src
    except:
        import sys
        import os.path as op
        sys.path.append(op.dirname(op.dirname(op.dirname(op.dirname(__file__)))))
        sys.path.append(op.dirname(op.dirname(__file__)))

def path_valid_check(path: str, format: list) -> bool:
    try:
        os.path.isdir(path)
    except:
        raise AssertionError("wrong path provided")
    try:
        path[-3:] in format or path[-4:] in format
    except:
        raise InvalidFileFormatException(item=path[-4:])
