import copy
import os
from pathlib import Path

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from OCC.Core.StlAPI import StlAPI_Writer

from amworkflow.geometry import simple_geometries, composite_geometries


class Geometry: # TODO datastore as parent class??
    def __init__(self, parameters: dict) -> None:
        """general geometry class


        """

        self.p = copy.deepcopy(parameters)

    @staticmethod
    def parameter_description(self) -> dict[str,str]:
        """description of the parameters needed for the centerline model

            to be overwritten by user
        """

        description = {
            "name": "name of the design",
            "out_dir": "directory where the output files are stored",

            "stl_linear_deflect": "?? default = ??",
            "stl_angular_deflect": "?? default = ??",

            "TODO": "TODO",
        }
        return description

    def geometry_spawn(self) -> TopoDS_Shape:
        """define geometry using occ fct or own

            to be overwritten by user
        """

        pass

    def create(self) -> None:
        """ create step and stl file from the geometry

            fixed method
        """

        print('in create')
        # create step file
        self.shape = self.geometry_spawn()

        # export as stl and step file
        assert "out_dir" in self.p
        stl_output_dir = Path(self.p["out_dir"])
        assert stl_output_dir.is_dir()


        outfile = stl_output_dir / f"{self.p['name']}.stl"

        # TODO
        stl_write = StlAPI_Writer()
        stl_write.SetASCIIMode(True)  # Set to False for binary STL output
        status = stl_write.Write(self.shape, self.p["name"])
        print(status)
        if status:
            print("Done!")

        # # default stl parameters
        # self.p["stl_linear_deflection"] = self.p.get("stl_linear_deflection", 0.001)
        # self.p["stl_angular_deflection"] = self.p.get("stl_angular_deflection", 0.1)
        # write_stl_file(self.shape, outfile, mode="binary",
        #                linear_deflection=self.p["stl_linear_deflection"],
        #                angular_deflection=self.p["stl_angular_deflection"],
        #                )

        # not working
        # outfile = stl_output_dir / f"{self.p['name']}.stp"
        # write_step_file(a_shape=self.shape, filename=outfile)

        pass



class GeometryCenterline(Geometry):
    def __init__(self, parameters: dict) -> None:
        """geometry class for centerline model

        creating a layer by layer geometry from a given array of centerline points (x,y,z)

        """
        super().__init__(parameters)

    @staticmethod
    def parameter_description(self) -> dict[str,str]:
        """description of the parameters needed for the centerline model
        """
        description = {
            "name": "name of the design",
            "points": "array with the centerline points in the format x,y,z",
            "layer_thickness": "thickness of the layers",
            "number_of_layers": "number of layers",
            "layer_height": "height of the layers",
        }

        return description

    def geometry_spawn(self) -> TopoDS_Shape:
        """define geometry from centerline points
        """
        # where to put those geometry building function?
        # is_close as global parameter?
        # wall_maker = CreateWallByPointsUpdate(self.p["name"], self.p["layer_thickness"], self.p["layer_height"]*self.p["number_of_layers"], is_close=False)
        # design = wall_maker.Shape()
        design = None
        return design

class GeometryParamWall(Geometry):
    def __init__(self, parameters: dict) -> None:
        """geometry class for parametric wall element

        """

        super().__init__(parameters)

    @staticmethod
    def parameter_description(self) -> dict[str,str]:
        """description of the parameters needed for the centerline model
        """
        description = {
            "name": "name of the design",
            "height": "array with the centerline points in the format x,y,z",
            "length": "length of wall",
            "width": "width of wall",
            "radius": "radius of wall curvature",
            "infill": "infill of wall: possible solid, honeycomb, zigzag",
            "layer_thickness": "thickness of the layers",
            "number_of_layers": "number of layers",
            "layer_height": "height of the layers",
        }

        return description

    def geometry_spawn(self) -> TopoDS_Shape:
        """define geometry
        """
        print('in spawn of child class')
        if self.p["infill"] == "solid":
            box = simple_geometries.create_box(length=self.p["length"],
                                     width=self.p["width"],
                                     height=self.p["height"],
                                     radius=self.p["radius"],
                             )
            print(box)
            return box
