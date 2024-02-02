[![tests](https://github.com/BAMresearch/amworkflow/actions/workflows/tests.yml/badge.svg)](https://github.com/BAMresearch/amworkflow/actions/workflows/tests.yml)

## Motivation

This repository contains a module for creating automated workflows in the context of concrete additive manufacturing.

### Workflow steps
* design definition (class Geometry)
* create mesh file (class Meshing)
* generate GCODE (class Gcode)
* perform FEM simulation (of print process or final printed structure) (class Simulation)

### folder structure
* src/amworkflow: source code of the amworkflow package
* tests: pytest for general routines
* examples: example usecases
* usecases: new usecases


## Conda
```conda env create -f environment.yml```

## Installation
activate the environment if you did not:
```bash
conda env create -f environment.yml
conda activate amworkflow
```

to the root directory and then

```bash
doit install
```
Alternatively,
* you can do it manually:
    First clone the required lib:
    ```bash
    git clone https://github.com/tpaviot/pythonocc-utils.git
    ```
    Install it:
    ```bash
    pip install ./pythonocc-utils
    ```
    Last step, install amworkflow locally:
    ```bash
    pip install -e .
    ```

Then you are good to go.

## Explore examples
Example workflows can be found in folder examples. 
Run them by calling the doit file in the subfolder. 
```bash
cd examples/<example_name>
doit -f dodo_<example_name>.py
```

### Toy
The toy example is a simple example to show the basic workflow steps.

### Wall [![wall](https://github.com/BAMresearch/amworkflow/actions/workflows/wall.yml/badge.svg)](https://github.com/BAMresearch/amworkflow/actions/workflows/wall.yml)
The workflow is created for a curved wall element with geometrical parameters like length, thickness, width and height with different infill structures.

### TrussArc [![trussarc](https://github.com/BAMresearch/amworkflow/actions/workflows/trussarc.yml/badge.svg)](https://github.com/BAMresearch/amworkflow/actions/workflows/trussarc.yml)
A arc with truss structure is given by a list of points defining the centerline.
The design is created by those points and additional parameters like layer thickness and the gcode and simulation is set-up.


## Create new usecase  

By
```bash
doit -s new_case case_name="<name>"
```
a new folder under /usecases will be created with the name given containing a template workflow as basis for the new usecase.

Or copy the template.py file and use it as your new usecase.



