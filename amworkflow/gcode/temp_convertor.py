def command(line: str):
    '''Write one line of command to gcode file
    '''
    return line+"\n"

def move(x: float, y: float, z: float, e: float, f: float):
    '''Move to a point in space
    '''
    return command(f"G1 X{x} Y{y} Z{z} E{e} F{f}")


