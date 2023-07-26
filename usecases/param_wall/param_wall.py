from amworkflow.src.core.workflow import BaseWorkflow
from amworkflow.src.geometries.simple_geometry import create_box

class ParamWallWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()

    def geometry_spawn(self, param):
        pm = param
        box = create_box(length=pm.length,
                         width= pm.width,
                         height=pm.height,
                         radius=pm.radius)
        return box


p_wall = ParamWallWorkflow()
# print(cli())
p_wall.create()
p_wall.mesh()
p_wall.download()