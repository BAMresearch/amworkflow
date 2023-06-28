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
Doit command not available for now. Testing the codes by running /usecases/param_wall/param_wall.py. Results are stored first in /amworkflow/src/infrastructure/database/files/output_files with hashcode names and then downloaded with human-readable names under the specific usecase folder.
