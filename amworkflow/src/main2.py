from src.interface.sub_window import td_visualizer
from src.geometries.simple_geometry import create_box, create_cylinder
from src.utils.writer import stl_writer
from src.constants.enums import Directory
print(Directory.SYS_PATH.value)
box = create_box(30, 20, 10)
stl_writer(box, "new_brick.stl")
td_visualizer(box)
# td_visualizer(create_cylinder(0.6, 3))
