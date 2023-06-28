from dolfinx.io import  XDMFFile
from dolfinx import mesh
from dolfinx.fem import FunctionSpace, Function, locate_dofs_topological, dirichletbc, Constant, petsc, form, assemble_scalar
from mpi4py import MPI

comm = MPI.COMM_WORLD
encoding=XDMFFile.Encoding.HDF5
file = "/home/yuxiangpt/docs/BAM-Project/amworkflow/amworkflow/src/infrastructure/database/files/output_files/7abcad0bbd3249eeabc121a353ea88f5.xdmf"
with XDMFFile(comm,file,'r',encoding=encoding) as ifile:
   domain = ifile.read_mesh(name="19c75b5f57504a5887c060d43d23a83f")
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
V = FunctionSpace(domain, ("CG", 1))
uD = Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = locate_dofs_topological(V, fdim, boundary_facets)
bc = dirichletbc(uD, boundary_dofs)
import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
from petsc4py.PETSc import ScalarType
f = Constant(domain, ScalarType(-6))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
V2 = FunctionSpace(domain, ("CG", 2))
uex = Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
L2_error = form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")
import pyvista
# print(pyvista.global_theme.jupyter_backend)
from dolfinx import plot
pyvista.start_xvfb()
print("xvfb starts successfully")
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
print("plotter starts successfully")
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    print("start plotter")
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")