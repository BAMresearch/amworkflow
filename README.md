## Motivation

This repository contains a module for creating automated workflows in the context of concrete additive manufacturing.

### Workflow steps
* parameter definition (class parameter)
* design definition (class geometry)
* create mesh file (class mesh)
* generate GCODE (class gcode)
* perform FEM simulation (of print process or final printed structure) (class simulation)

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
    python -m pip install .
    ```

Then you are good to go.

## Create new usecase  

By
```bash
doit new_case -n <name>
```
a new folder under /usecases will be created with the name given containing a template workflow as basis for the new usecase.

Or copy the template.py file and use it as your new usecase.



