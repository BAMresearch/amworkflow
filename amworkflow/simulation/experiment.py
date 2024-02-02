import dolfinx as df
import numpy as np
import pint
import ufl
from dolfinx.io import XDMFFile
from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.boundary_conditions.boundary import (
    plane_at,
    point_at,
    within_range,
)
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.util import Parameters, QuadratureRule, ureg
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

"""As base class for experiments the Experiment class of FenicsXConcrete is used.
    Attention pint parameters are required!"""


class ExperimentProcess(Experiment):
    """Experiment class for AM process simulations.

    geometry by external mesh file
    boundary conditions: fixed on bottom (z==0), am body force in z directory
    """

    def __init__(self, parameters: dict[str, pint.Quantity] | None = None) -> None:
        """initializes the object, for the rest, see base class

        Standard parameters are set
        setup function called

         Args:
            parameters: dictionary with parameters that can override the default values. Needs to be pint quantities!! description is given in parameter_description()

        """

        # initialize a set of default parameters
        p = Parameters()

        p.update(parameters)

        super().__init__(p)

    def setup(self) -> None:
        """Generates the mesh by loading the mesh file given in parameter"""

        if self.p["dim"] == 3:
            # load mesh file
            with XDMFFile(MPI.COMM_WORLD, self.p["mesh_file"], "r") as xdmf:
                domain = xdmf.read_mesh(name="Grid")
                # change to base units
                nodes = domain.geometry.x[:] * ureg(self.p["mesh_unit"])
                domain.geometry.x[:] = nodes.to_base_units().magnitude
                self.mesh = domain

        else:
            raise ValueError(
                f"wrong dimension: {self.p['dim']} is not implemented for problem setup"
            )

    @staticmethod
    def parameter_description() -> dict[str, str]:
        """description of the required parameters for the experiment """

        description = {
            "dim": "dimension of problem, only 3D implemented",
            "mesh_file": "path name of the mesh file in xdmf format (h5 with same name required)",
            "mesh_unit": "unit of the mesh file for recomputing into base units",
        }

        return description

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """sets up a working set of parameter values as example

        Returns:
            dictionary with a working set of the required parameter
        """

        default_parameters = {}

        # mesh information
        # dimension of problem, only 3D implemented
        default_parameters["dim"] = 3 * ureg("")
        # path name of the mesh file in xdmf format
        default_parameters["mesh_file"] = "bla/test_files.xdmf" * ureg("")
        # unit of the mesh file
        default_parameters["mesh_unit"] = "mm" * ureg("")

        return default_parameters

    def create_displacement_boundary(
        self, V: df.fem.FunctionSpace
    ) -> list[df.fem.bcs.DirichletBCMetaClass]:
        """defines displacement boundary as fixed at bottom

        Args:
            V: function space

        Returns:
            list of dirichlet boundary conditions

        """

        bc_generator = BoundaryConditions(self.mesh, V)

        # find boundary at bottom
        mesh_points = self.mesh.geometry.x
        min_z = mesh_points[:, 2].min()
        print("fix at z=", min_z)

        if self.p["dim"] == 3:
            # fix dofs at bottom
            bc_generator.add_dirichlet_bc(
                np.array([0.0, 0.0, 0.0], dtype=ScalarType),
                boundary=plane_at(min_z, 2),
                method="geometrical",
                entity_dim=self.mesh.topology.dim - 1,  # surface
            )
        else:
            raise ValueError(
                f"wrong dimension: {self.p['dim']} is not implemented for problem setup"
            )

        return bc_generator.bcs

    def get_bottom_area(self) -> float:
        fdim = self.mesh.topology.dim - 1
        bottom_facets = df.mesh.locate_entities(self.mesh, fdim, self.boundary_bottom())
        marked_facets = np.hstack([bottom_facets])
        marked_values = np.hstack([np.full_like(bottom_facets, 1)])
        sorted_facets = np.argsort(marked_facets)
        facet_tag = df.mesh.meshtags(
            self.mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
        )

        ds = ufl.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=facet_tag,
            metadata={"quadrature_degree": 2},
        )

        # not required - can be done by gcode!
        area = df.fem.assemble_scalar(df.fem.form(1.0 * ds(1)))

        return area

    def create_body_force_am(
        self, v: ufl.argument.Argument, q_fd: df.fem.Function, rule: QuadratureRule
    ) -> ufl.form.Form:
        """defines body force for am experiments

        element activation via pseudo density and incremental loading via parameter ["load_time"] computed in class concrete_am

        Args:
            v: test_files function
            q_fd: quadrature function given the loading increment where elements are active
            rule: rule for the quadrature function

        Returns:
            form for body force

        """
        print("in force")
        force_vector = np.zeros(self.p["dim"])
        force_vector[-1] = -self.p["rho"] * self.p["g"]  # works for 2D and 3D
        print(force_vector)

        f = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = q_fd * ufl.dot(f, v) * rule.dx

        return L


class ExperimentStructure(Experiment):
    """Experiment class for AM structure simulation in the end (after printing).

    geometry by external mesh file (z direction == vertical printing direction (layer height))
    boundary conditions from testing controlling via parameter ["bc_setting"]
            fixed_y_bottom: y_min surface fixed and displacement load at y_max surface
    with or without body force param["rho"] (in y direction)
    #TODO: generate sufficient loading cases parameterized!
    """

    def __init__(self, parameters: dict[str, pint.Quantity] | None = None) -> None:
        """initializes the object, for the rest, see base class

        Standard parameters are set
        setup function called

         Args:
            parameters: dictionary with parameters that can override the default values, description is given in parameter_description()

        """

        # initialize a set of default parameters
        p = Parameters()

        p.update(parameters)

        super().__init__(p)

        # initialize variable top_displacement
        self.top_displacement = df.fem.Constant(
            domain=self.mesh, c=0.0
        )  # applied via fkt: apply_displ_load(...)

    def setup(self) -> None:
        """Generates the mesh loading the mesh file given in parameter"""

        if self.p["dim"] == 3:
            # load mesh file
            with XDMFFile(MPI.COMM_WORLD, self.p["mesh_file"], "r") as xdmf:
                domain = xdmf.read_mesh(name="Grid")
                # change to base units
                nodes = domain.geometry.x[:] * ureg(self.p["mesh_unit"])
                domain.geometry.x[:] = nodes.to_base_units().magnitude
                self.mesh = domain
        else:
            raise ValueError(
                f"wrong dimension: {self.p['dim']} is not implemented for problem setup"
            )

    @staticmethod
    def parameter_description() -> dict[str, str]:
        """description of the required parameters for the experiment """

        description = {
            "bc_setting": "boundary setting, possible cases <fixed_y_bottom> fixed at ymin values loaded per displacement load at ymax values",
            "dim": "dimension of problem, only 3D implemented",
            "mesh_file": "path name of the mesh file in xdmf format (h5 with same name required)",
            "mesh_unit": "unit of the mesh file for recomputing into base units",
        }

        return description

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """sets up a working set of parameter values as example

        Returns:
            dictionary with a working set of the required parameter
        """

        default_parameters = {}

        # boundary setting
        default_parameters["bc_setting"] = "fixed" * ureg("")  # boundary setting
        # mesh information
        # dimension of problem, only 3D implemented
        default_parameters["dim"] = 3 * ureg("")
        # path name of the mesh file in xdmf format
        default_parameters["mesh_file"] = "bla/test_files.xdmf" * ureg("")
        # unit of the mesh file
        default_parameters["mesh_unit"] = "mm" * ureg("")

        return default_parameters

    def create_displacement_boundary(
        self, V: df.fem.FunctionSpace
    ) -> list[df.fem.bcs.DirichletBCMetaClass]:
        """Defines the displacement boundary conditions

        Args:
            V: Function space of the structure

        Returns:
            list of DirichletBC objects, defining the boundary conditions

        """

        # define boundary conditions generator
        bc_generator = BoundaryConditions(self.mesh, V)

        if self.p["bc_setting"] == "fixed_y_bottom":
            if self.p["dim"] == 3:
                # normally geometries are defined as x/y and z in height due to AM production here y is the height!!
                # get mesh_points to define boundaries
                mesh_points = self.mesh.geometry.x
                min_y = mesh_points[:, 1].min()
                max_y = mesh_points[:, 1].max()

                # min_x = mesh_points[:, 0].min()
                # max_x = mesh_points[:, 0].max()
                # min_z = mesh_points[:, 2].min()
                # max_z = mesh_points[:, 2].max()

                print("fix at y=", min_y)
                print("check points at y_min", mesh_points[mesh_points[:, 1] == min_y])
                # dofs at bottom y_min fixed in x, y and z direction
                bc_generator.add_dirichlet_bc(
                    np.array([0.0, 0.0, 0.0], dtype=ScalarType),
                    boundary=plane_at(min_y, 1),
                    method="geometrical",
                    entity_dim=self.mesh.topology.dim - 1,  # surface
                )
                # dy = 0.001
                # bc_generator.add_dirichlet_bc(
                #     np.array([0.0, 0.0, 0.0], dtype=ScalarType),
                #     boundary=within_range(
                #         [min_x, min_y - dy, min_z], [max_x, min_y + dy, max_z]
                #     ),
                #     method="geometrical",
                #     entity_dim=self.mesh.topology.dim - 2,  # points
                # )

                # top displacement at top (==max_y) in y direction
                # look for max y values in mesh
                print("apply disp where", max_y)
                print("check points at max_y", mesh_points[mesh_points[:, 1] == max_y])
                bc_generator.add_dirichlet_bc(
                    self.top_displacement,
                    boundary=plane_at(max_y, 1),
                    sub=1,
                    method="geometrical",
                    entity_dim=self.mesh.topology.dim - 1,
                )
                # dy = 0.001
                # bc_generator.add_dirichlet_bc(
                #     np.array([0.0, 0.0, 0.0], dtype=ScalarType),
                #     boundary=within_range(
                #         [min_x, max_y - dy, min_z], [max_x, max_y + dy, max_z]
                #     ),
                #     method="geometrical",
                #     entity_dim=self.mesh.topology.dim - 2,  # points
                # )

        else:
            raise ValueError(f"Wrong boundary setting: {self.p['bc_setting']}")

        return bc_generator.bcs

    def apply_displ_load(self, top_displacement: pint.Quantity | float) -> None:
        """Updates the applied displacement load

        Args:
            top_displacement: Displacement of the top boundary in mm

        """
        top_displacement.ito_base_units()
        self.top_displacement.value = top_displacement.magnitude

    # def create_body_force(self, v: ufl.argument.Argument) -> ufl.form.Form:
    #     """defines body force
    #
    #     Args:
    #         v: test_files function
    #
    #     Returns:
    #         form for body force
    #
    #     """
    #
    #     force_vector = np.zeros(self.p["dim"])
    #     force_vector[1] = (
    #         -self.p["rho"] * self.p["g"]
    #     )  # for 3D case where y is the height direction!!
    #
    #     f = df.fem.Constant(self.mesh, ScalarType(force_vector))
    #     L = ufl.dot(f, v) * ufl.dx
    #
    #     return L
