import sys
import os
import pathlib
from OCC.Core.TopoDS import TopoDS_Shape
from amworkflow.src.core.workflow import BaseWorkflow
from amworkflow.src.geometries.simple_geometry import create_box
from amworkflow.src.interface.cli.cli_workflow import cli

class ParamWallWorkflow(BaseWorkflow):
    def __init__(self, args):
        super().__init__(args)

    def geometry_spawn(self, param) -> TopoDS_Shape:
        pm = param
        box = create_box(length=pm.length,
                         width= pm.width,
                         height=pm.height,
                         radius=pm.radius)
        return box


p_wall = ParamWallWorkflow(cli())
# print(cli())
# p_wall.create()
# p_wall.mesh()
# p_wall.download()