import gmsh
import logging
from amworkflow.src.constants.exceptions import GmshUseBeforeInitializedException

def mesh_visualizer():
    try:
        gmsh.is_initialized()
    except:
        raise GmshUseBeforeInitializedException()