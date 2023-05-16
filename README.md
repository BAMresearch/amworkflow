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

