import gmsh
import logging

def mesh_visualizer():
    try:
        gmsh.is_initialized()
    except:
        logging.info("Gmsh must be initialized first!")
    gmsh.fltk.run()