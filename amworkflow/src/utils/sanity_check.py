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

def path_append_check():
    try:
        import src
    except:
        import sys
        import os.path as op
        sys.path.append(op.dirname(op.dirname(op.dirname(op.dirname(__file__)))))
        sys.path.append(op.dirname(op.dirname(__file__)))