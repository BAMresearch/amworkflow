## Motivation

This repository contains a module for creating automated workflows in the context of concrete additive manufacturing.

### Workflow
* parametrized design (defined via CAD program)
* generate stl file 
* generate GCODE --> printer
* generate mesh file
* generate FEM simulation

### folder structure
* amworkflow: general routines
* tests: pytest for general routines
* usecases: example usecases


## Conda
```conda env create -f environment.yml```

## Installation
After creating the environment using conda, one library have to be installed mannually since neither Pypi nor Conda has the distribution.

```git clone https://github.com/tpaviot/pythonocc-utils.git```

to the root directory and then

```pip install ./pythonocc-utils```

## How to use
This is by far the most user-friendly version. To create a new workflow, simply do:
```python
from amworkflow.src.interface.api import amWorkflow as aw
@aw.engine.amworkflow
def geometry_spawn(pm):
    box = aw.geom.create_box(length=pm.length,
                        width= pm.width,
                        height=pm.height,
                        radius=pm.radius)
    return box
```
The function name *geometry_spawn* and form parameter *pm* are only examples, however it is recommended to stick with these names in case of overwriting functions in the *Baseworkflow*. 

TODO: builtin functions in the future updates will have "_" before their names to avoid such conflict.

FIXME: The *BaseWorkflow* is still under heavy modification hence not working for now.

