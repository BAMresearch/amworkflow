import argparse
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gp", "--geom_param", nargs="+")
    parser.add_argument("-gpv", "--geom_param_value", nargs="+", type=float)
    parser.add_argument("-ip", "--iter_param", nargs="*", type=int)
    parser.add_argument("-mbl", "--mesh_by_layer", nargs="?", type=float)
    parser.add_argument("-mbt", "--mesh_by_thickness", nargs="?", type=float)
    parser.add_argument("-msf", "--mesh_size_factor", nargs="?", type=float)
    parser.add_argument("-stlad", "--stl_angular_deflect", nargs="?", type=float)
    parser.add_argument("-stlld", "--stl_linear_deflect", nargs="?", type=float)
    parser.add_argument("-n", "--name", nargs="?", type=str)
    parser.add_argument("-y", "--yaml_dir", nargs="?", type=str)
    parser.add_argument("-impt", "--import_dir", nargs="?", type=str)
    parser.add_argument("-e", "--edit", action="store_true")
    parser.add_argument("-rm", "--remove", action="store_true")
    parser.add_argument("-f", "--file", nargs="?", type=str)
    args = parser.parse_args()
    return args

