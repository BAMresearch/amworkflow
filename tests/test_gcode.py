import os

from amworkflow.gcode.main import Gcode


def test_gcode():
    gcode = Gcode()
    gcode.width = 8
    gcode.height = 3
    gcode.nozzle_diameter = 8
    gcode.feedrate = 10
    gcode.layer_height = 2
    gcode.layer_num = 1
    gcode.create("test", "test")
    try:
        os.remove("test")
    except FileNotFoundError:
        print(f"File 'test' not found.")
    print("".join(gcode.gcd_writer.gcode))
    assert (
        "".join(gcode.gcd_writer.gcode)
        == "G90\nM82\nM106 S0\nM104 S0\nT0\nG1 Z0 F10\nG1 X2 Y4 E0 F10\nG1 X3 Y5 E0.6752\nG1 X4 Y6 E0.6752\nG1 X5 Y7 E0.6752\nM104 S0\nM107\nM140\nM84\n"
    )
