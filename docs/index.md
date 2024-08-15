# AMworkflow

# Dev on Documentation
+ Prerequisite: ```Sphinx```, ```sphinx-autobuild```, ```sphinx-rtd-theme```, ```sphinxcontrib-applehelp```, ```sphinxcontrib-devhelp```, ```sphinxcontrib-htmlhelp```, ```sphinxcontrib-jquery```, ```sphinxcontrib-jsmath```, ```sphinxcontrib-qthelp```, ```sphinxcontrib-serializinghtml```.
+ How does it work?
Use the command:

```sphinx-autobuild . _build/html```

The server should be ready on Localhost Port 8000 and it auto builds when changes made.


## Introduction
Some Intro

## Project Organization
Some info about organization

## BuiltinCAD

### Shortest_distance_point_line
![Small dist line point](pics/Pasted-Graphic.png)
The function returns the shortest distance between a point and a line (segment). For example in figure above the result will be $(d_{min},\frac{L_1}{L_1+L_2})$.

```{eval-rst}
.. autofunction:: amworkflow.geometry.builtinCAD.shortest_distance_point_line
```




```{math}
e^{i\pi} + 1 = 0
```

```{eval-rst}
.. automodule:: amworkflow.geometry.builtinCAD
   :members: CreateWallByPoints
```
