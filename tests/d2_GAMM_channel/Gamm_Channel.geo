mesh_size = 0.25;
Point(1) = {0, 0, 0, mesh_size};
Point(2) = {0.5, 0, 0, mesh_size};
Point(3) = {1.5, 0, 0, mesh_size};
Point(4) = {0, 0.1, 0, 0.2};
Point(5) = {0, 1, 0, 0.2};
Point(6) = {1.5, 1, 0, mesh_size};
Line(1) = {4, 5};
Line(2) = {5, 6};
Line(3) = {6, 3};
Line(4) = {3, 2};
Point(7) = {0, -1.2, 0, 1.0};
Circle(5) = {2, 7, 4};
Line Loop(6) = {5, 1, 2, 3, 4};
Plane Surface(7) = {6};

Symmetry {1, 0, 0, 0} {
  Duplicata { Line{1, 5, 4, 3, 2}; }
}
Point(9) = {0, 0, 0, mesh_size};
Point(10) = {0.5, 0, 0, mesh_size};
Point(11) = {1.5, 0, 0, mesh_size};
Line Loop(13) = {9, 1, 12, 11, 10};
Plane Surface(14) = {-13};

//Mesh.Algorithm = 2;
//Mesh.Algorithm = 5; //Delaunay
Mesh.Algorithm = 6; //Frontal
Mesh.RecombineAll = 1;
Mesh.Smoothing = 5;
Mesh.SubdivisionAlgorithm = 1;

// Bottom
Physical Line(1) = {10, 4};
// Bump
Physical Line(2) = {9, 5};
// Top
Physical Line(3) = {12, 2};
// Inlet
Physical Line(4) = {11};
// Outlet
Physical Line(5) = {3};
Physical Surface(19) = {14, 7};

