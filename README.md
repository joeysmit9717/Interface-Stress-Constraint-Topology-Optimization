# Interface-Stress-Constraint-Topology-Optimization
This project covers the application of an Interface Stress Constraint in a multi-material topology optimization setting. Although the interface strength varies with the method of fabrication, the paper presents an indiscriminate method for handling angle dependent interface stresses.

## Dataset & Attribution

The dataset used in this project was obtained from its original source and is not owned or created by me. It is used for educational and portfolio purposes only.

All rights and ownership of the original dataset remain with their respective owners. Please refer to the original source for licensing information and terms of use.

What this folder contains
-------------------------

This package contains the documentation and environments for producing the 
computations and visualization for the (angle dependent) interface stress constraint in
a multi-material (2 materials) topology optimization setting where limits are set on tensile stresses within a material interface. 
The test setting remains constant:
96 x 48 elements and forces acting on the 7 right-most nodes on the bottom of the 
component. The weaker material (material 1) has a Young's modulus of 1 MPa and the stronger material
(material 2) has a Young's modulus of 2 MPa.

  - **README.md**          - General overview of the project
  - **InterfaceStressConstraint.pdf**          - Report on the findings
  - **Python codes**  - codes for producing the plots in the report
  
Key Findings
------------

While the compliance converged to similar values for both the angle-independent and the angle-dependent settings,
the final designs show clear differences in the placement of the material interface. The blue space is the weaker material (1 MPa)
and the red material is the stronger one (2 MPa). 


![Final Design (angle independent)](Report/ISC17min240FinalDesign.png)


Python codes
------------
  - **Helmholtz2materialsISC17Forces.py**        - Angle independent version
  - **Helmholtz2materialsAngleForces.py**       - Angle dependent version
  - **Helmholtz2materialsISC17vsAngle.py**     - Produces the compliance comparison plot


What the analysis does
----------------------

1. Adds an Interface Stress Constraint (ISC) to a multi-material topology
  optimization setting.
2. ISC split up in two version: constant stress limits within an interface and
   an angle dependent version.
3. Compares the compliance optimization trajectory for both cases and compares them.
4. Visualizes final results in 2D and 3D plots and provides clear plots on the interfaces.
   

How to regenerate the plots
---------------------------

-  Helmholtz2materialsISC17Force.py    - Plots associated with angle independent ISC
-  Helmholtz2materialsISCangleForces.py           - Plots associated with angle dependent ISC
-  Helmholtz2materialsISC17vsAngle.py   - Compliance history comparison


Requirements
------------
Python 3.x with: numpy, scipy, matplotlib, nlopt, joblib, and line_profiler.

Key result snapshot 
-----------------------------------

- Compliance remained similar when switching from an angle-independent to an angle-dependent ISC.
 
- Final designs are visibly different, as interface areas are moved around to adapt to local stresses

- code is optimized, but still takes a considirable amount of time to run. Nel_x at line 41 in Helmholtz2materialsISC17Forces.py sets
  the amount of elements on the x-axis. While 24, 48 and 96 are easy to run, 96 elements takes several hours to complete.

- For Nel_x set to 192, it is necessary to have at least 32 GB of system RAM to handle the large temporary datasets.
