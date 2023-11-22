## Short introduction on mechanics of the G-Code

G-Code controls filament flows by manipulating two parameters, E and F.

$E$ stands for “Extrude” which defines a coordinate of the material (G92 defines the absolute coordinate system and G91, the relative.)

$F$ stands for “Feed Rate”. Although its literal meaning seems to be the rate of feeding, well it actually specifies the moving speed of the printer head. The subtle differences will be explained below.

```G-Code
G1 X0.4 Y0.6 E0.04 F1000
```

```G-Code
G1 E0.04 F1000
```

Two code snippets above leads to different results. 

* In the first example, the machine first interpolates the travel from the previous location (let's say (0,0)) to the location specified. The travel time can be calculated hence. Then the machine allocates the material whose amount is defined by parameter $E$ uniformly on the travel. The figure below depicts the aforementioned process.

    ![plot1](imgs/plot_relation_EF.jpeg)

    There are two speed values, $f_1$, $f_2$ (slope)here. The principle of command G1 is to make sure that,

    $$\frac{l}{f_2} \stackrel{!}{=} \frac{e}{f_1}$$

    $f_1$ here is the command $F$ used in the G-Code. Therefore, to have a workaround to control $f_1$, the $E$ can be given various values to achieve that.

    Travel Time $T$ here can be calculated by,

    $$T = \frac{l}{f_2}$$

* In the second example, the machine directly interpolatese the consumption of filaments by $F$. Therefore, $F$ here is the speed $f_2$ in the previous example. 

    Travel Time $T$ here can be calculated by,

    $$T = \frac{e}{f_1}$$

## Assumptions

Assuming no residue, no inconsistent printing and the printer is well calibrated, the printing process can be treated as a transition from a cylinder to a prism with conservation of the volume. 

Let $A$ be the area of the nozzle; 

$L$ be the travel length of the printer head;

$M$ be the mass of filament consumed;

$V$ be the volume of filament consumed;

$V_w$ be the volume of water consumed;

$V_c$ be the volume of concrete consumed;

$w,h$ be the width and height of the printing line;

$E$ be the extraction between the current coordinate and the previous one of the filament ($\delta E$);

$\rho$ be the density of the material;

$\rho_c$ be the density of the concrete material;

$\rho_w$ be the density of the water;

$M_c$ be the mass of the concrete material;

$M_w$ be the mass of the water.
## Model

* For the first example:
    Travel Length:
    $$L = \|(X_1-X_2,Y_1-Y_2)\|_2$$
    Travel Time:
    $$T = \frac{L}{F}$$
    Volume:
    $$V = E \cdot A = L\cdot w \cdot h $$
    Discharge:
    $$Q = \frac{V}{T} = E \cdot A \cdot \frac{F}{L}= w \cdot h \cdot F$$
    Density of the material:
    $$\rho = \frac{M_c+\rho_w V_w}{\frac{M_c}{\rho_c}+V_w}$$
    
* For the second example:

    Travel Time:
    $$T = \frac{E}{F}$$
    Density of the material:
    $$\rho = \frac{M_c+\rho_w V_w}{\frac{M_c}{\rho_c}+V_w}$$
    Volume:
    $$V = E \cdot A = \frac{M}{\rho}$$
    Discharge:
    $$Q = \frac{V}{T} = A\cdot F$$



## Correction Coefficient

The models above are established on the condition that the printer is well calibrated, suggesting that value E reflects the real material consumption. When it is not, the calibration will be necessary.

The second model will be adopted to find the correction coefficient $\kappa$.

Consequently, The Volume Equation can be modified as,

$$V = E_r \cdot A = \frac{M}{\rho}$$
$$\kappa = \frac{E_r}{E}$$

where $E_r$ is the value E rectified by coefficient $\kappa$.


## Data
### Data collected from experiments.

Three groups of experiments are conducted, with parameters F, E varing from 5 to 10. The mass of material extruded is recorded in each time and labeled as M1, M2 and M3, representing mass in different times. 
```csvtable
source: PrinterCalibExperimentEValue_unit.csv
```

## Calculation
```csvtable
columns:
- Mean_M
- theoretical_V
- theoretical_D
- E
- rectified_E
- Er
source: output_file.csv
```
## Result
### Correction Coefficients
The Correction Coefficients, denoted as $\kappa$, calculated from three groups of experiments are 300, 253, and 128. Notably, during the experimentation with parameters in the third group, the concrete experienced stiffening. Consequently, data from M2 and M3 in this group is considered invalid as the extruder could no longer properly extrude materials. It is worth noting that the initial data point, M1: 103, is identified as an outlier. As a result, the outcomes derived from M1 are also considered unreliable.

Ideally, the material extruded in the second group should be precisely half the amount of that in the first group. However, in the experiment, this value is observed to be 16% less. The shortfall is attributed to the stiffening process, which impedes the provision of materials and subsequent extrusion. Given the significant impact of stiffening, it is advisable to consider only the data from the first group as valid.

### Estimated Width
Since there is no chance to conduct an experiment based on mode 1, the only way of testing the model is to use example gcode. The parameters and corresponding geometry parameters this gcode uses are:

| F(mm/min) | width(mm) | layer height(mm) | A($mm^{2}$) | 
| --------- | --------- | ---------------- | ----------- |
| 1800      | 9.95      | 3                |      50.265       |

The coordination of the first position: 
$$X29.943, Y148.839$$
The next position:
$$X30.005, Y149.157$$
E is:
$$\delta E = 0.0003$$
Hence,
$$L = \sqrt{0.0062^{2}+0.318^{2}}= 0.318$$
$$E_{rectified}=0.0003\cdot300=0.09$$
$$w = E_{rectified}\cdot A\cdot h\cdot L = 0.09\cdot 50.265\cdot 3\cdot 0.318=4.31$$
The width calculated based on $\kappa=300$ is 56% less. However it is questionable that the width provided by the example G-Code is accurate, and whether the material is of the same composition.
## Conclusion
Given the invalidity of data in Group 3 and the questionable nature of Group 2, it is recommended to consider the data from Group 1 as valid. The calculated Correction Coefficient ($\kappa$​) from Group 1 is determined to be 300.

It is also recommended to conduct another group of experiments which print actual line on the printing bed, aiming for calibrating the line width and test the validity of the correction coefficient in the meantime. 