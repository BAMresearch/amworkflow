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

## How to use
for now, using
``` doit start ```
will run the GUI interface, displaying geometries and storing them sequentially. These functions of the command will be altered in the future developments.
