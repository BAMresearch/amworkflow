from OCC.Core.TopoDS import TopoDS_Shape
from amworkflow.geometry.composite_geometries import CreateWallByPointsUpdate

class Geometry(): # TODO datastore as parent class??
    def __int__(self, parameters: dict) -> None:
        """general geometry class


        """

        self.p = parameters

    @staticmethod
    def parameter_description(self) -> dict[str,str]:
        """description of the parameters needed for the centerline model

            to be overwritten by user
        """

        description = {
            "name": "name of the design",
            "out_dirc": "directory where the output files are stored",

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
        # create step file
        self.shape = self.geometry_spawn()

        # export as stl saved in self.p["out_dir"]
        self.write_stl()

        # export as step file
        self.write_step()
        pass

    def write_stl(self):
        """write stl file from occ shape

        """
        # from old aw.tool
        # self.shape, self.p["stl_linear_deflect"], self.p["stl_angular_deflect"]
        pass

    def write_step(self):
        """write step file from occ shape

        """
        # from old aw.tool
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
        wall_maker = CreateWallByPointsUpdate(self.p["name"], self.p["layer_thickness"], self.p["layer_height"]*self.p["number_of_layers"], is_close=False)
        design = wall_maker.Shape()

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
        if "infill" == "solid":
            box = create_box(length=self.p["length"],
                                     width=self.p["width"],
                                     height=self.p["height"],
                                     radius=self.p["radius"],
                             )
            return box


    return