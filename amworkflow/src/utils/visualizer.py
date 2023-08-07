import gmsh
import logging
from amworkflow.src.constants.exceptions import GmshUseBeforeInitializedException

def mesh_visualizer():
    try:
        gmsh.is_initialized()
    except:
        raise GmshUseBeforeInitializedException()
    gmsh.fltk.run()
    
def color_background(value):
    if value:
        return 'background-color: green'
    # else:
    #     return 'background-color: blue'