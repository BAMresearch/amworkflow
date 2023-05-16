// Gmsh project
// creating a simple wall
// small test version
// stl = 0; // set to 1 or 0
DefineConstant[ length = 60]; // mm
DefineConstant[ width = 6];   // mm
DefineConstant[ height = 8]; // mm
DefineConstant[ layer_height = 1 ]; // mm
DefineConstant[ meshSize = 1];
// // more realistic dimensions
// length = 3000; // mm
// width = 200;   // mm
// height = 2000; // mm
// meshSize = 200;
//
number_layer = height/layer_height;
//
// outline XY Plane Z==0
Point(1) = {0, 0, 0, meshSize};
Point(2) = {length, 0, 0, meshSize};
Point(3) = {length, width, 0, meshSize};
Point(4) = {0, width, 0, meshSize};
//+
Line(1) = {4, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
//+
Curve Loop(1) = {2, 3, 4, 1};
//+
Plane Surface(1) = {1};
//+

//
// Surface export (STL)
If(stl > 0)
  // Extrusion total height
  Extrude {0, 0, height} {Surface{1};}
  //
  // Physical Surface
  Physical Surface("1", 28) = {13, 26, 17, 1, 25, 21};  //
  // Mesh 2D for stl export
  // export MESH STL per "Physical Surface"
Else
  //Extrusion per layer
  Extrude {0, 0, layer_height} {Surface{1}; } // -> look for surface number
  ID_start = 26; // number of first top level surface
  idx = 22; // from testing delta between surface numbers
  For k In {2:number_layer}
    Extrude {0, 0, layer_height} {
    Surface{ID_start+(k-2)*idx};
    }
  EndFor
  //+
  Physical Volume("1", 27) = {1:number_layer};
  Physical Surface("1", 28) = {1};  // bottom boundary condition
  // Mesh 3D for mesh export (Fenics, vtk)
  //Transfinite Surface "*";
  //Recombine Surface "*";
  //Transfinite Volume "*";
  //
EndIf
//+
//+
