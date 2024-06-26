import logging
import multiprocessing
import typing
from pathlib import Path

import gmsh
import meshio
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from OCC.Core.TopoDS import TopoDS_Solid
from OCC.Extend.DataExchange import read_step_file

from amworkflow import occ_helpers

typing.override = lambda x: x

from amworkflow.config.settings import LOG_LEVEL

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("amworkflow.meshing.meshing")
# logger.setLevel(LOG_LEVEL)


class Meshing:
    """Base class with API for any meshing."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    @typing.override
    def create(
        self, step_file: Path, out_xdmf: Path, out_vtk: Path | None = None
    ) -> None:
        """Create mesh xdmf file and possibly vtk file from input step file.

        Args:
            step_file: File path to input geometry step file.
            out_xdmf: File path of output xdmf file (maybe others if other mesh engine).
            out_vtk: Optional file path of output vtk file.

        Returns:

        """
        raise NotImplementedError


class MeshingGmsh(Meshing):

    def __init__(
        self,
        mesh_size_factor: float = 0.1,
        layer_height: float | None = None,
        number_of_layers: float | None = None,
        **kwargs,
    ) -> None:
        """Meshing class using gmsh.

        Args:
            mesh_size_factor: Factor for mesh size (default 0.1).
            layer_height: Height of each layer (default None).
            number_of_layers: Number of layers (default None).
        """
        self.mesh_size_factor = mesh_size_factor
        self.layer_height = layer_height
        self.number_of_layers = number_of_layers
        super().__init__(**kwargs)

    def create(
        self, step_file: Path, out_xdmf: Path, out_vtk: Path | None = None
    ) -> None:
        """Create mesh xdmf file and possibly vtk file from input step file.

        Args:
            step_file: File path to input geometry step file.
            out_xdmf: File path of output xdmf file.
            out_vtk: Optional file path of output vtk file.

        Returns:

        """
        if self.layer_height is None and self.number_of_layers is None:
            raise ValueError("Both layer_height and number_of_layers are None.")
        assert step_file.is_file(), f"Step file {step_file} does not exist."

        shape = read_step_file(filename=str(step_file))
        solid = occ_helpers.create_solid(shape)

        assert isinstance(solid, TopoDS_Solid), "Must be TopoDS_Shape object to mesh."

        gmsh.initialize()

        try:
            gmsh.is_initialized()
        except:
            raise EnvironmentError("Gmsh not initialized.")

        # two options of splitting geometry in layers checked above in assert
        geo = occ_helpers.split_by_plane(
            item=shape, layer_height=self.layer_height, nz=self.number_of_layers
        )

        model = gmsh.model()
        gmsh.option.setNumber("General.Verbosity", 0)
        threads_count = multiprocessing.cpu_count()
        # gmsh.option.setNumber("General.NumThreads", threads_count) # FIX: Conflict with doit. Will looking for solutions.
        # model.add("model name") # TODO: required? Not necessarily but perhaps for output .msh
        layers = model.occ.importShapesNativePointer(int(geo.this), highestDimOnly=True)
        model.occ.synchronize()
        for layer in layers:
            model.add_physical_group(3, [layer[1]], name=f"layer{layer[1]}")
            # phy_gp = model.getPhysicalGroups()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", self.mesh_size_factor)
        model.mesh.generate()
        model.mesh.remove_duplicate_nodes()
        model.mesh.remove_duplicate_elements()
        phy_gp = model.getPhysicalGroups()
        model_name = model.get_current()

        # # save
        # msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
        # msh.name = model_name
        # cell_markers.name = f"{msh.name}_cells"
        # facet_markers.name = f"{msh.name}_facets"
        # with XDMFFile(msh.comm, out_xdmf, "w") as file:
        #     file.write_mesh(msh)
        #     file.write_meshtags(cell_markers)
        #     msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        #     file.write_meshtags(facet_markers)

        # workaround for since gmshio.model_to_mesh is not working
        out_msh = out_xdmf.with_suffix(".msh")
        gmsh.write(str(out_msh))
        msh = meshio.read(out_msh)
        mesh = self.create_mesh(msh, "tetra")
        meshio.write(out_xdmf, mesh)

        if out_vtk:
            gmsh.write(str(out_vtk))

    def create_mesh(self, mesh, cell_type: str, prune_z: bool = False) -> meshio.Mesh:
        """Convert meshio mesh to fenics compatible mesh.
            based on https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html?highlight=read_mesh

        Args:
            mesh: Mesh read by meshio from msh file (meshio.read(file_msh)).
            cell_type: Type of cell to be meshed (e.g. 'tetra','triangle' ...).
            prune_z: True for 2D meshes - removes z coordinate.

        Returns:
            out_mesh: Mesh which can be saved as xdmf file. (meshio.write(file_xdmf, out_mesh))
        """
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )
        return out_mesh
