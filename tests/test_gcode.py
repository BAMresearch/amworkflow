import os
from pathlib import Path

from amworkflow.gcode.gcode import GcodeFromPoints


def test_gcode(tmp_path):
    gcd = GcodeFromPoints()
    caller_path = Path(os.path.dirname(__file__))
    file_point = caller_path.parent / "examples" / "RandomPoints" / "RandomPoints.csv"
    file_gcode = tmp_path / "RandomPoints.gcode"
    gcd.create(file_point, file_gcode)
    assert file_gcode.exists()
