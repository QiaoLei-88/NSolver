//
//  BndNaca4DigitSymm::test.cpp
//  NSolver
//
//  Created by 乔磊 on 15/9/14.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#include <NSolver/BoundaryManifold/BndNaca4DigitSymm.h>

namespace NSFEMSolver
{
  using namespace dealii;

  void BndNaca4DigitSymm::test() const
  {
    const double Pi = std::atan (1.0) * 4.0;
    std::ofstream foil_gmsh ("NACA0012.geo");
    std::ofstream solve_out ("solve_test.txt");
    foil_gmsh.precision (8);
    int point_counter = 0;
    for (int i=100; i>0; --i)
      {
        const double x = 1.0 - std::cos (static_cast<double> (i)/200.0 * Pi);
        const double y = thickness (x);
        const double weight = std::sin (static_cast<double> (i)/100.0 * Pi);
        foil_gmsh << "Point(" << ++point_counter << ") = {"
                  << x << ", " << y << ", 0.0, "
                  << 0.1* (1.0-weight) + weight << "*mesh_foil_size};\n";
      }
    for (int i=0; i<=100; ++i)
      {
        const double x = 1.0 - std::cos (static_cast<double> (i)/200.0 * Pi);
        const double y = -thickness (x);
        const double weight = std::sin (static_cast<double> (i)/100.0 * Pi);
        foil_gmsh << "Point(" << ++point_counter << ") = {"
                  << x << ", " << y << ", 0.0, "
                  << 0.1* (1.0-weight) + weight << "*mesh_foil_size};\n";
      }

    foil_gmsh << "\nSpline(1) = { 1";
    for (int i=2; i<=point_counter; ++i)
      {
        foil_gmsh << ", " << i;
      }
    foil_gmsh << "};" << std::endl;

    solve_out <<   "             x        solved_x      solved_x-x"
              << "          foil_x       solved_fx    solved_fx-fx"
              << "          foil_y       solved_fy    solved_fx-fy\n";
    solve_out << std::scientific;
    solve_out.precision (8);
    // {
    //   const double x = 1.0 - std::cos (1.0/200.0 * Pi);
    //   Fad_db x_ad = x;
    //   x_ad.diff (0,1);

    //   const double x_foil = x_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
    //   const double y_foil = y_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

    //   const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

    //   Fad_db sx_ad = solved_x;
    //   sx_ad.diff (0,1);
    //   const double solved_fx = x_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   const double solved_fy = y_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
    //             << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
    //             << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
    //             << std::endl;
    // }
    // {
    //   const double x = 1.0 - std::cos (88.0/200.0 * Pi);
    //   Fad_db x_ad = x;
    //   x_ad.diff (0,1);

    //   const double x_foil = x_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
    //   const double y_foil = y_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

    //   const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

    //   Fad_db sx_ad = solved_x;
    //   sx_ad.diff (0,1);
    //   const double solved_fx = x_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   const double solved_fy = y_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
    //             << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
    //             << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
    //             << std::endl;
    // }
    // {
    //   const double x = 1.0 - std::cos (1.0/200.0 * Pi);
    //   Fad_db x_ad = x;
    //   x_ad.diff (0,1);

    //   const double x_foil = x_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
    //   const double y_foil = y_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

    //   const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

    //   Fad_db sx_ad = solved_x;
    //   sx_ad.diff (0,1);
    //   const double solved_fx = x_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   const double solved_fy = y_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
    //             << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
    //             << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
    //             << std::endl;
    // }
    // {
    //   const double x = 1.0 - std::cos (88.0/200.0 * Pi);
    //   Fad_db x_ad = x;
    //   x_ad.diff (0,1);

    //   const double x_foil = x_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
    //   const double y_foil = y_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

    //   const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

    //   Fad_db sx_ad = solved_x;
    //   sx_ad.diff (0,1);
    //   const double solved_fx = x_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   const double solved_fy = y_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
    //   solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
    //             << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
    //             << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
    //             << std::endl;
    // }
    foil_gmsh.close();
    solve_out.close();
    return;
  }

}
