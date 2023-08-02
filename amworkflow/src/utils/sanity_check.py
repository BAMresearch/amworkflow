from amworkflow.src.constants.exceptions import DimensionViolationException, NoDataInDatabaseException, InvalidFileFormatException
import numpy as np
import os

def import_freecad_check():
    """
     @brief Check if freecad is installed and if so import FreeCAD. This is needed to avoid importing FREECAD in order to be able to run a part program that is not available on the system.
     @return True if freecad is installed False if not or error occurs during import of the freecad
    """
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
    """
     @brief Checks if the dimensions are correct. This is a helper function to make sure that the dimensions are in the correct order
     @param dm_list List of dimensions
    """
    # Checks the width and length of the list of DM items.
    for dm_item in dm_list:
        # Check if the dimensions of the item are within the radius.
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
    """
     @brief Append path to sys. path if source files are not available. This is necessary to avoid import errors
    """
    try:
        import src
    except:
        import sys
        import os.path as op
        sys.path.append(op.dirname(op.dirname(op.dirname(op.dirname(__file__)))))
        sys.path.append(op.dirname(op.dirname(__file__)))

def path_valid_check(path: str, format: list = None) -> bool:
    """
     @brief Check if path is valid and return file name if not raise InvalidFileFormatException. This is used to check if file can be read from file system
     @param path path to file or directory
     @param format list of file formats to check if file is in
     @return filename's extension or False if file is not in format ( no extension ) or file is not in
    """
    split_result = path.rsplit("/", 1)
        # split_formt = split_result[]
        # Split the result of a split command.
    if len(split_result) > 1:
        dir_path, filename = split_result
    if os.path.isdir(dir_path) == False:
        raise AssertionError("wrong path provided")
    if format != None:
        
        fmt = filename.rsplit(".",1)[1]
        # if dir_path is not a directory
        
        # If filename is not in format raise InvalidFileFormatException item filename
        if fmt not in format:
            raise InvalidFileFormatException(item=fmt)
        else:
            return fmt
    else:
        return True