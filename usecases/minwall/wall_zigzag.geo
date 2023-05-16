// Gmsh project
DefineConstant[ stl = 1]; // DEFINED VIA DOIT FILE !!!//-> extrusion by total height and physical surface for 2D mesh export
// stl=0 //-> extrusion by layer height and physical volume for 3D mesh export
//
// wall with zigzag infill
//
// small example DEFINED VIA DOIT FILE !!!
DefineConstant[ length = 60 ]; // mm
DefineConstant[ width = 6 ];   // mm
DefineConstant[ height = 8 ]; // mm
DefineConstant[ layer_width = 2 ]; // mm
DefineConstant[ layer_height = 1 ]; // mm
DefineConstant[ meshSize = 1 ];
// more realistic dimensions
// length = 3000; // mm
// width = 200;   // mm
// height = 40; //2000; // mm
// layer_width = 50; // mm
// layer_height = 10; // mm
// meshSize = 10;
//
number_layer = height/layer_height;
// fixed zig zag line with 3 elements
//  4-----------------------------------------------------3
//  |   8-----------10--11--------------------------13-7  |
//  |  |              /17\                           / 14 |
//  |   9  /                          \12/             |  |
//  |   5-18-------------------------10-15-------------6  |
//  1-----------------------------------------------------2
//
// outer outline XY Plane Z==0
Point(1) = {0, 0, 0, meshSize};
Point(2) = {length, 0, 0, meshSize};
Point(3) = {length, width, 0, meshSize};
Point(4) = {0, width, 0, meshSize};
//+
Line(1) = {4, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
// inner outline XY Plane Z==0
Point(5) = {layer_width, layer_width, 0, meshSize};
Point(6) = {length-layer_width, layer_width, 0, meshSize};
Point(7) = {length-layer_width, width-layer_width, 0, meshSize};
Point(8) = {layer_width, width-layer_width, 0, meshSize};
//+
//Line(5) = {8, 5};
//Line(6) = {5, 6};
//Line(7) = {6, 7};
//Line(8) = {7, 8};
//+
// Curve Loop(1) = {2, 3, 4, 1};
// Curve Loop(2) = {6, 7, 8, 5};
// zigzag infill fixed dritteln
hh = length-2*layer_width;
hh1 = hh/3; // dritteln
bb = width-2*layer_width;
cc = Sqrt(hh1*hh1 + bb*bb);
cos_alpha = hh1/cc;
sin_alpha = bb/cc;
dy = 0.5*layer_width/cos_alpha;
dx = 0.5*layer_width/sin_alpha;
Point(9) = {layer_width, layer_width+dy, 0, meshSize};
Point(10) = {layer_width+hh1-dx, width-layer_width, 0, meshSize};
Point(11) = {layer_width+hh1+dx, width-layer_width, 0, meshSize};
Point(12) = {layer_width+2*hh1, layer_width+dy, 0, meshSize};
Point(13) = {layer_width+3*hh1-dx, width-layer_width, 0, meshSize};
Point(14) = {layer_width+hh, width-layer_width-dy, 0, meshSize};
Point(15) = {layer_width+2*hh1+dx, layer_width, 0, meshSize};
Point(16) = {layer_width+2*hh1-dx, layer_width, 0, meshSize};
Point(17) = {layer_width+hh1, width-layer_width-dy, 0, meshSize};
Point(18) = {layer_width+dx, layer_width, 0, meshSize};
//
//+
Line(5) = {9, 8};
Line(6) = {8, 10};
Line(7) = {10, 9};
//
Line(8) = {18, 16};
Line(9) = {16, 17};
Line(10) = {17, 18};
//
Line(11) = {11, 13};
Line(12) = {13, 12};
Line(13) = {12, 11};
//
Line(14) = {15, 14};
Line(15) = {14, 6};
Line(16) = {6, 15};
//+
Curve Loop(1) = {2, 3, 4, 1}; //outer line
Curve Loop(2) = {5, 6, 7}; //inner lines
Curve Loop(3) = {8, 9, 10}; //inner lines
Curve Loop(4) = {11, 12, 13}; //inner lines
Curve Loop(5) = {14, 15, 16}; //inner lines
//+
Plane Surface(1) = {1, 2, 3, 4, 5};

//
// Surface export (STL)
If(stl > 0)
  // Extrusion total height
  Extrude {0, 0, height} {Surface{1};}
  //
  // Physical Surface
  Physical Surface("1", 99) = {49, 37, 41, 98, 45, 1, 57, 85, 65, 73, 89, 81, 77, 97, 69, 53, 61, 93};
  //
  // Mesh 2D for stl export
  // export MESH STL per "Physical Surface"
Else
  //Extrusion per layer
  Extrude {0, 0, layer_height} {Surface{1}; } // -> look for surface number
  ID_start = 98; // number of first top level surface (only ring 50)
  idx = 82; // from testing delta between surface numbers (only ring 42)
  For k In {2:number_layer}
      Extrude {0, 0, layer_height} {
      Surface{ID_start+(k-2)*idx};
      }
  EndFor
  //+
  Physical Volume("1", 27) = {1:number_layer};
  Physical Surface("1", 28) = {1}; // Bottom
  // Mesh 3D for mesh export (Fenics, vtk)
  //
EndIf
//+
//+

