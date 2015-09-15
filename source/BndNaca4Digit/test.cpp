


#include <NSolver/BoundaryManifold/BndNaca4Digit.h>

namespace NSFEMSolver
{
  using namespace dealii;

  void BndNaca4Digit::test() const
  {
    const double Pi = std::atan (1.0) * 4.0;
    std::ofstream camber_out ("NACA_camber.txt");
    std::ofstream thickness_out ("NACA_thickness.txt");
    std::ofstream foil_out ("NACA_foil.txt");
    std::ofstream solve_out ("solve_test.txt");
    for (int i=100; i>0; --i)
      {
        const double x = 1.0 - std::cos (static_cast<double> (i)/200.0 * Pi);
        Fad_db x_ad = x;
        x_ad.diff (0,1);
        double x_foil = x_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        double y_foil = y_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        foil_out  << x_foil << "\t" << y_foil << std::endl;
      }

    for (int i=0; i<=100; ++i)
      {
        const double x = 1.0 - std::cos (static_cast<double> (i)/200.0 * Pi);
        Fad_db x_ad = x;
        x_ad.diff (0,1);

        double x_foil = x_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        double y_foil = y_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        foil_out  << x_foil << "\t" << y_foil << std::endl;
        camber_out << x  << "\t" << camber (x_ad).val() << std::endl;
        thickness_out << x << "\t" << thickness (x) << std::endl;
      }

    solve_out <<   "             x        solved_x      solved_x-x"
              << "          foil_x       solved_fx    solved_fx-fx"
              << "          foil_y       solved_fy    solved_fx-fy\n";
    solve_out << std::scientific;
    solve_out.precision (8);
    {
      const double x = 1.0 - std::cos (2.0/200.0 * Pi);
      Fad_db x_ad = x;
      x_ad.diff (0,1);

      const double x_foil = x_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
      const double y_foil = y_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

      const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

      Fad_db sx_ad = solved_x;
      sx_ad.diff (0,1);
      const double solved_fx = x_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      const double solved_fy = y_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
                << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
                << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
                << std::endl;
    }
    {
      const double x = 1.0 - std::cos (88.0/200.0 * Pi);
      Fad_db x_ad = x;
      x_ad.diff (0,1);

      const double x_foil = x_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
      const double y_foil = y_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

      const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

      Fad_db sx_ad = solved_x;
      sx_ad.diff (0,1);
      const double solved_fx = x_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      const double solved_fy = y_lower<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
                << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
                << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
                << std::endl;
    }
    {
      const double x = 1.0 - std::cos (2.0/200.0 * Pi);
      Fad_db x_ad = x;
      x_ad.diff (0,1);

      const double x_foil = x_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
      const double y_foil = y_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

      const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

      Fad_db sx_ad = solved_x;
      sx_ad.diff (0,1);
      const double solved_fx = x_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      const double solved_fy = y_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
                << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
                << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
                << std::endl;
    }
    {
      const double x = 1.0 - std::cos (88.0/200.0 * Pi);
      Fad_db x_ad = x;
      x_ad.diff (0,1);

      const double x_foil = x_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
      const double y_foil = y_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));

      const double solved_x = solve_parameter (Point<2> (x_foil, y_foil));

      Fad_db sx_ad = solved_x;
      sx_ad.diff (0,1);
      const double solved_fx = x_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      const double solved_fy = y_upper<double> (solved_x, std::atan (camber (sx_ad).fastAccessDx (0)));
      solve_out << x << "  " << solved_x << "  " << solved_x - x << "  "
                << x_foil << "  " << solved_fx << "  " << solved_fx - x_foil << "  "
                << y_foil << "  " << solved_fy << "  " << solved_fy - y_foil << "  "
                << std::endl;
    }

    camber_out.close();
    thickness_out.close();
    foil_out.close();
    solve_out.close();
    return;
  }

}
