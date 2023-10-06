from OCC.Core.TopoDS import TopoDS_Shape

class Geometry(): # TODO datastore as parent class??
    def __int__(self, parameters: dict) -> None:
        """general geometry class


        """

        self.p = parameters

    def geometry_spawn(self) -> TopoDS_Shape:
        """define geometry using occ fct or own

            to be overwritten by user
        """
        self.shape = None

        pass
    def create(self) -> None:
        """ create step and stl file from the geometry
        """

        pass


class GeometryCenterline(Geometry):
    def __init__(self, parameters: dict) -> None:
        """geometry class for centerline model

        """
        super().__init__(parameters)

    def geometry_spawn(self) -> TopoDS_Shape:
        """define geometry using occ fct or own

            to be overwritten by user
        """
        self.shape = None

        pass
    def create(self) -> None:
        """ create step and stl file from the geometry
        """

        pass