## Conda
```conda env create -f environment.yml```

## Installation
After creating the environment using conda, one library have to be installed mannually since neither Pypi nor Conda has the distribution.
```bash
git clone https://github.com/tpaviot/pythonocc-utils.git
```
activate the environment if you did not:
```bash
conda activate amworkflow
```

to the root directory and then

```bash
pip install ./pythonocc-utils
```

Last step, install amworkflow locally:
```bash
python -m pip install .
```

Then you are good to go.

## Get started
This is a simple tutorial which get you familiar with the way of interacting with amworkflow. 
1. Create a new folder in /usecase, for example test_am4.
2. Create a new script am4.py. (You may also find it in examples/test_am4.)
3. simply do:
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

4. In your terminal move to the directory:
    ```bash
    cd usecases/test_am4
    ```
    Then run:
    ```bash
    python am4.py -n p_wall -gp length width height radius -gpv 1 2 3 10 -ip 2 4 2 4 20 4 -mbt 0.2 -msf 0.1
    ```
    It means, 
    | Command | Description |
    | --- | --- |
    | -n | creating a new model profile with model name "p_wall" |
    |-gp| and geometry parameters *length, width, height, radius* (as are corresponding with the ones in the *pm* in *am4.py*. The order doesn't matter.) |
    |-gpv|input geometry parameters correspondingly, which are 1, 2, 3 and 10|
    |-ip|Setting iteration parameters by the index number, endpoint and number of geom_files expected. In this example, we iterate parameter *width(2)* from its start point which is defined in -gp, to its endpoint(4) and we want 2 geom_files including the startpoint and endpoint.Also we want iterate *radius*(4) from 10 to 20 with 4 geom_files created in total.|
    |-mbt|Mesh geometries with layer thickness defined. In this case every layer will be 0.2 of thickness.|
    |-msf|Mesh geometries with a global size factor. It is recommended to set a number smaller than the layer thickness to get a better performance.|

    The workflow will firstly creates a subfolder /db and /output. /db contains a database file dedicated for this usecase, and all files created by the workflow which are stored in its subfolder /files. /output contains all files auto downloaded by the workflow and files downloaded manually by users.

    It will take a while for generating geometries and meshes. Once files are generated successfully, the workflow will automatically download them to the folder /output with its task number as the folder name, and all files organized.

5. You would probably see somthing like this if step 4 goes well:
    ```bash
    ...
    Info    : No ill-shaped tets in the mesh :-)
    Info    : 0.00 < quality < 0.10 :         0 elements
    Info    : 0.10 < quality < 0.20 :         0 elements
    Info    : 0.20 < quality < 0.30 :         0 elements
    Info    : 0.30 < quality < 0.40 :       635 elements
    Info    : 0.40 < quality < 0.50 :       908 elements
    Info    : 0.50 < quality < 0.60 :      1614 elements
    Info    : 0.60 < quality < 0.70 :      4200 elements
    Info    : 0.70 < quality < 0.80 :      9377 elements
    Info    : 0.80 < quality < 0.90 :     12712 elements
    Info    : 0.90 < quality < 1.00 :      6084 elements
    Info    : Done optimizing mesh (Wall 1.42704s, CPU 1.4269s)
    Info    : 95095 nodes 619224 elements
    Info    : Removing duplicate mesh nodes...
    Info    : Found 0 duplicate nodes 
    Info    : No duplicate nodes found
    Info    : Removing duplicate mesh elements...
    Info    : Done removing duplicate mesh elements
    Task230809184748 Downloaded successfully.
    ```

    now if you run:
    ```bash
    python am4.py
    ```

    The workflow will return infomation of this usecase:
    ```bash
    Model Profile:
            created_date imported_file_id model_name
    0 2023-08-09 18:47:47             None     p_wall
    Model Parameters:
            created_date param_name
    0 2023-08-09 18:47:47     length
    1 2023-08-09 18:47:47      width
    2 2023-08-09 18:47:47     height
    3 2023-08-09 18:47:47     radius
    Parameter Properties:
    param_status model_name param_name        created_date
    0          True     p_wall     length 2023-08-09 18:47:47
    1          True     p_wall      width 2023-08-09 18:47:47
    2          True     p_wall     height 2023-08-09 18:47:47
    3          True     p_wall     radius 2023-08-09 18:47:47
    Imported Files:
    Empty DataFrame
    Columns: []
    Index: []
    Task:
    model_name   stl  xdmf    vtk       task_id    stp    h5    msh
    0     p_wall  True  True  False  230809184748  False  True  False
    ```

    You could already see the model profile you just created and the task you ran. 

    If you want to run another task with the same set of parameters, it is unnecessary to use -gp to define the geometry_paremeters again. If you forget the parameters or its order, simply type:
    ```bash
    python am4.py -n p_wall
    ```
    The workflow will return the paremeters and its order:
    ```bash
    Activated Parameter(s) of model p_wall: ['length', 'width', 'height', 'radius']
    ```
    then you can run:
    ```bash
    python am4.py -n p_wall -gpv 5 7 9 20
    ```

    The workflow will return something like:
    ```
    no mesh file in task 230809190418 found, skip...
    Task230809190418 Downloaded successfully.
    ```

6. What if you need more parameters or different parameters?
    amorkflow supports changing parameters by adding a param_status in the db. In this example, let's change our parameter *radius* to *radius_1* and add a new parameter *length_0*. To achieve that, firstly you need to change parameters in the am4.py:
    ```python
    from amworkflow.src.interface.api import amWorkflow as aw
    @aw.engine.amworkflow()
    def geometry_spawn(pm):
        box = aw.geom.create_box(length=pm.length,
                            width= pm.width,
                            height=pm.height,
                            radius=pm.radius_1)
        return box
    ```
    Should any parameter be changed or deleted, it can no longer show up in this script until you employ it again. Then run:
    ```bash
    python am4.py -n p_wall -e -gp length length_0 width height radius_1
    ```
    You should see the return:
    ```
    Deactivate parameter(s):['radius'] 
    Activate parameter(s): ['length_0', 'radius_1'] 
    Add new parameter(s): ['length_0', 'radius_1']
    ```
    then you can run a task to test it:
    ```bash
    python am4.py -n p_wall -gp length length_0 width height radius_1 -gpv 2 3 4 5 10
    ```
    then we get the info again:
    ```bash
    python am4.py
    ```
    The workflow will return something like:
    ```
    Model Profile:
         created_date imported_file_id model_name
    0 2023-08-09 18:47:47             None     p_wall
    Model Parameters:
    param_name        created_date
    0     length 2023-08-09 18:47:47
    1      width 2023-08-09 18:47:47
    2     height 2023-08-09 18:47:47
    3     radius 2023-08-09 18:47:47
    4   length_0 2023-08-09 19:14:50
    5   radius_1 2023-08-09 19:14:50
    Parameter Properties:
    param_status model_name param_name        created_date
    0          True     p_wall     length 2023-08-09 18:47:47
    1          True     p_wall      width 2023-08-09 18:47:47
    2          True     p_wall     height 2023-08-09 18:47:47
    3         False     p_wall     radius 2023-08-09 18:47:47
    4          True     p_wall   length_0 2023-08-09 19:14:50
    5          True     p_wall   radius_1 2023-08-09 19:14:50
    Imported Files:
    Empty DataFrame
    Columns: []
    Index: []
    Task:
    model_name   stl   xdmf    vtk       task_id    stp     h5    msh
    0     p_wall  True   True  False  230809184748  False   True  False
    1     p_wall  True  False  False  230809190418  False  False  False
    2     p_wall  True  False  False  230809191716  False  False  False
    ```
    You can see that model parameters won't be really deleted but be deactivated. Parameter *radius* is now given a False to its param_status and new parameters are activated by default.

    Now we change radius_1 in am4.py back to radius and run:
    ```bash
    python am4.py -n p_wall -e -gp length width height radius
    ```
    to change the model back, the return would be:
    ```
    Deactivate parameter(s):['radius_1', 'length_0'] 
    Activate parameter(s): ['radius'] 
    Add new parameter(s): []
    ```

For more details and usages of the workflow, check out in wiki.

## Motivation

This repository contains a module for creating automated workflows in the context of concrete additive manufacturing.

### Workflow
* parametrized design (defined via CAD program)
* generate geometry file (.stl, .stp)
* generate GCODE --> printer
* generate mesh file (.msh, .xdmf & .h5, .vtk)
* generate FEM simulation

### Built-in file-management
* Management of files generated by the workflow by a file-based database.
 

### folder structure
* amworkflow: general routines
* tests: pytest for general routines
* usecases: example usecases

