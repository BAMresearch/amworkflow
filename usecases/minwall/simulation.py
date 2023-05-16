import pathlib

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
from mpi4py import MPI

""" build up fenics simulation """


def amsimulation(
    param: dict[str, str | float],
    xdmf_mesh_file: pathlib.Path | str,
    gcode_file: pathlib.Path | str,
    xdmf_out_file: pathlib.Path | str,
) -> None:
    """
    simulation
    1. read mesh and create initial path
    2. am simulation using FenicsConcrete here just path update no computation

    Args:
        param: dict of simulation and geoemtry parameters
        xdmf_mesh_file: xdmf mesh file for fenics
        gcode_file: gcode file describing the tool path
        xdmf_out_file: result file (in the moment pseudo density over time)
    """

    # set up mesh and inital path variable
    # load mesh file
    with XDMFFile(MPI.COMM_WORLD, xdmf_mesh_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        # ct = xdmf.read_meshtags(mesh, name="Grid")

    # print(mesh.ufl_cell())
    # print(mesh.topology.dim)

    # subdomains for boundary conditions and layer area (required to get time per layer)
    def bottom(x):
        return np.isclose(x[2], 0)  # z==0

    fdim = domain.topology.dim - 1
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)

    marked_facets = np.hstack([bottom_facets])
    marked_values = np.hstack([np.full_like(bottom_facets, 1)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )

    ds = ufl.Measure(
        "ds", domain=domain, subdomain_data=facet_tag, metadata={"quadrature_degree": 2}
    )

    # not required - can be done by gcode!
    param["layer_area"] = fem.assemble_scalar(
        fem.form(1.0 * ds(1))
    )  # required for path creation
    # print('check A',param["layer_area"], param['length']*param["width"])

    # init path function and density
    # V = fem.FunctionSpace(domain, ("CG", 1))
    V = fem.FunctionSpace(domain, ("DG", 0))  # one value per element makes more sence
    path = fem.Function(V, name="path time")
    dens = fem.Function(V, name="pseudo density")

    path_out = define_path_time(path, param, gcode=gcode_file)

    # output file
    xdmf_out = XDMFFile(domain.comm, xdmf_out_file, "w")
    xdmf_out.write_mesh(domain)

    dt = param["dt"]  # simulation time step
    time = param["height"] / param["layer_height"] * param["time_layer"]
    t = 0
    while t <= time:
        # compute density based on path function
        dens_list = pd_fkt(path_out.x.array[:])
        dens.x.array[:] = dens_list[:]

        # export path & density into xdmf for t
        # dens.name = "pseudo density"
        # path_out.name = "path time"
        xdmf_out.write_function(dens, t)
        xdmf_out.write_function(path_out, t)

        # update path and time
        path_out.x.array[:] += dt
        t += dt
        # print(f'check new path for t: {t}:', path_out.x.array.min(), path_out.x.array.max())

    return


def define_path_time(
    path: fem.Function,
    param: dict[str, str | float],
    gcode: pathlib.Path | str | None = None,
) -> fem.Function:
    """create path as layer wise at dof points
     one layer by time
     if gcode == None without gcode information using global layer_height parameter
     if gcode == path using gcode information for element selection

    Args:
        path: initialized function for path
        param: dict of global parameters

    Returns:
        path function

    """
    length_layer = None
    if gcode:
        points_per_layer = read_gcode(gcode)
        points_layer0 = points_per_layer[0]
        # print("points layer 0", points_layer0)

        # compute layer length for time per layer
        length_layer = 0
        for i in range(len(points_layer0) - 1):
            length_layer += np.linalg.norm(points_layer0[i + 1] - points_layer0[i])
        print("length layer", length_layer)

    # additional parameters?
    T_0 = 0.0  # start time (default)

    # time for one layer # from velocity
    if not length_layer:
        # x-y plane / layer width ~ single length of one layer
        length_layer = param["layer_area"] / param["layer_width"]
        # otheriwse use computed layer length form gcode

    time_layer = length_layer / param["velocity"]

    # extend dict
    param["time_layer"] = time_layer
    layer_number = param["height"] / param["layer_height"]
    tol = 1e-5

    # dof map for coordinates
    dof_map = path.function_space.tabulate_dof_coordinates()
    print("CHECK", len(dof_map), len(path.x.array[:]), len(path.vector[:]))
    # print(dof_map)
    new_path = np.zeros(len(dof_map))

    x_CO = np.array(dof_map)[:, 0]
    y_CO = np.array(dof_map)[:, 1]
    z_CO = np.array(dof_map)[:, -1]

    h_min = np.arange(0, param["height"], param["layer_height"])
    h_max = np.arange(
        param["layer_height"],
        param["height"] + param["layer_height"],
        param["layer_height"],
    )

    # print(z_CO)
    # print(h_min)
    # print(h_max)

    for i in range(0, len(h_min)):
        layer_index = np.where((z_CO > h_min[i] - tol) & (z_CO <= h_max[i] + tol))
        new_path[layer_index] = (
            T_0 - (layer_number - 1 - i) * time_layer
        )  # negative time when layer will be reached
        # get elements from gcode points TODO

    # which one is correct? same len as dofmap -> in the moment all the same!
    path.vector[:] = new_path  # overwrite
    path.x.scatter_forward()
    # path.x.array[:] = new_path[:] # in x also GP included

    return path


def read_gcode(gcode: pathlib.Path | str) -> list:
    """

    Args:
        gcode: gcode file

    Returns:
        list including the x/y points for each layer in gcode files
    """
    # read gcode as txt to identify required lines
    layer_rows = []
    g1_rows = []
    with open(gcode, "r") as reader:
        all_lines = reader.readlines()

    for idx, line in enumerate(all_lines):
        if "z" in line.split(";")[0].lower():  # before comments
            layer_rows.append(idx)
        elif "g1" in line.split(";")[0].lower():
            g1_rows.append(idx)

    layer_rows.append(len(all_lines) - 1)  # number of line totally

    # print("rows where layer starts", layer_rows)
    # print("row with G1 command", g1_rows)
    g1_rows = np.array(g1_rows)

    # read x / y points per layer
    layer_points = []
    for layer_idx in range(len(layer_rows) - 1):
        # lines where points are given
        important_rows_idx_start = np.where(g1_rows > layer_rows[layer_idx])[0][0]
        important_rows_idx_end = np.where(g1_rows < layer_rows[layer_idx + 1])[0][-1]
        # print(important_rows_idx_start, important_rows_idx_end)
        important_rows_idx = np.arange(
            important_rows_idx_start, important_rows_idx_end + 1
        )
        # print("check", important_rows_idx)

        # print(f"important rows layer {layer_idx}:", important_rows_idx)
        # extract points
        x_values = []
        y_values = []
        for row_idx in important_rows_idx:
            splitted = all_lines[g1_rows[row_idx]].split(" ")
            for pp in splitted:
                if "x" in pp.lower():
                    x_values.append(float(pp.replace("X", "")))
                elif "y" in pp.lower():
                    y_values.append(float(pp.replace("Y", "")))

        points = np.array([x_values, y_values]).transpose()
        layer_points.append(points)
        # print("points", points)

    return layer_points


def pd_fkt(path_time: np.ndarray | list) -> np.ndarray | list:
    # pseudo density
    """

    :param path_time: vector of path time function
    :return:
    """
    tol = 1e-5

    l_active = np.zeros_like(path_time)
    activ_idx = np.where(path_time >= 0 - tol)[0]
    l_active[activ_idx] = 1.0  # active

    return l_active
