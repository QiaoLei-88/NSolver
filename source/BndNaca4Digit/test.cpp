


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
    camber_out.close();
    thickness_out.close();
    foil_out.close();
    return;
  }

}
