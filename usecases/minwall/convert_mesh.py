import pathlib

import meshio

"""convert msh file in fenics mesh"""


def convert_msh2xdmf(msh_file: pathlib.Path | str) -> None:
    """read msh file and convert to 3D tetra mesh for FeniCs

    Args:
        msh_file: file path to msh file
    """
    #

    msh = meshio.read(msh_file)
    print(msh)

    mesh = create_mesh(msh, "tetra")
    meshio.write(msh_file.with_suffix(".xdmf"), mesh)


def create_mesh(mesh, cell_type: str, prune_z: bool = False) -> meshio.Mesh:
    # convert msh file from https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html?highlight=read_mesh
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh
