

class BaseGeometry(object):
    def __init__():
        pass
    
    def data_assign():
        '''
        assign data pertained to the required geometry
        '''
        pass
    
    def geom_create_pygmsh():
        '''
        create the real entity of the geometry by Pygmsh.
        '''
        pass
    
    def geom_create_pyocc():
        '''
        create the real entity of the geometry by Pygmsh.
        '''
        pass
    
    def geom_create():
        '''
        create the real entity of the geometry by selecting the 
        desired CAD engine. The default engine would be Pygmsh.
        '''
        pass
    
    def geom_mesh():
        '''
        mesh the geom created by geom_create()
        '''
    
    def geom_slice():
        '''
        Slice the geometry created by geom_create()
        '''
        pass
    
    def geom_gcode():
        '''
        Create G-code for geometries sliced by geom_slice()
        '''
        pass
    