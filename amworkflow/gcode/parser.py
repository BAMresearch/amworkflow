import re
from pathlib import Path

from amworkflow.geometry import builtinCAD as bcad


class GCodeParser:
    def __init__(self, gcode: str):
        self.gcode_path = Path(gcode)
        self.tool_number = 0
        self.coordinate_system = 0
        self.fan_switch = 0
        self.fan_speed = 0
        self.motor_speed = 0
        self.data = []
        self.gcode_reader()

    def gcode_reader(self):
        counter = 0
        with self.gcode_path.open(encoding="utf-8") as f:
            print(f)
            for line in f:
                print(line)
                if line.startswith(";"):
                    continue
                self.data.append(self.line_reader(line))
                yield line

    def line_reader(self, line: str):
        """Read one line of gcode command

        :param line: One line of gcode command
        :type line: str
        :return: A dictionary of the values and index
        :rtype: dict
        """
        values = self.separator(line)
        head = line[0]
        return (head, values)

    def separator(self, line: str):
        """Separate the letter and value from the G-code line

        :param line: One line of G-code command
        :type line: str
        :return: A dictionary of the letters and values
        :rtype: dict
        """
        matches = re.findall(r"([GXYEMSFT])(\d+(\.\d+)?)", line)
        result = {
            key: int(value) if key in ("G", "M", "T") else float(value)
            for key, value, _ in matches
        }
        return result


gcodeparser = GCodeParser(
    "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/amworkflow/gcode/example.gcode"
)
print(gcodeparser.data)
