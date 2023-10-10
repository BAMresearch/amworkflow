import gmsh
from pathlib import Path



class Meshing(): # TODO datastore as parent class??
    def __int__(self, step_file, parameters) -> None:
        """general meshing class
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

            "mesh_size_factor": "mesh size factor for the meshing",
            "by_number_of_layers": "meshing by number of layers (True or False)",

            "TODO": "TODO",
        }
        return description
    def create(self, step_file: str|Path.Path) -> None:
        """ create xdmf file and vtk file
        """
        # read step file
        #shape = read shape from step_file

        # mesh by number of layers or layer height
        # model = self.mesher(...)

        # store mesh in xdmf file

        pass

    def mesher(self, shape: TopoDS_Shape,
               model_name: str,
               layer_type: bool,
               layer_param: float = None,
               size_factor: float = 0.1):
        """Meshing function for a given TopoDS_Shape using gmsh

            from old mesher.py
            """
        try:
            gmsh.is_initialized()
        except:
            raise GmshUseBeforeInitializedException()
        if not isinstance(shape, TopoDS_Solid):
            raise Exception("Must be Solid object to mesh.")
        if layer_type:
            geo = split(item=item,
                        split_z=True,
                        layer_thickness=layer_param)
        else:
            geo = split(item=item,
                        split_z=True,
                        nz=layer_param)
        model = gmsh.model()
        threads_count = psutil.cpu_count()
        gmsh.option.setNumber("General.NumThreads", threads_count)
        model.add(model_name)
        v = get_geom_pointer(model, geo)
        model.occ.synchronize()
        for layer in v:
            model.add_physical_group(3, [layer[1]], name=f"layer{layer[1]}")
            phy_gp = model.getPhysicalGroups()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
        model.mesh.generate()
        model.mesh.remove_duplicate_nodes()
        model.mesh.remove_duplicate_elements()
        phy_gp = model.getPhysicalGroups()
        model_name = model.get_current()
        return model

# if we have someday different meshing tools we can separte that into different meshing classes
# For now use Meshing class as the only one