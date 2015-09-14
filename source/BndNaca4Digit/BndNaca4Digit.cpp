


#include <NSolver/BoundaryManifold/BndNaca4Digit.h>

namespace NSFEMSolver
{
  using namespace dealii;

  BndNaca4Digit::BndNaca4Digit (const unsigned int number,
                                const double chord_length_in)
    :
    max_thickness (static_cast<double> (number%100)/100.0),
    max_camber (static_cast<double> ((number/1000)%10)/100.0),
    position_of_max_camber (static_cast<double> ((number/100)%10)/10.0),
    chord_length (chord_length_in)
  {}

  virtual
  Point<2>
  get_new_point_on_line (const typename Triangulation<2,2>::line_iterator &line) const
  {
    const Point<2> candidate = (line->vertex (0) + line->vertex (1)) / 2.0;
    const double r = solve_parameter (candidate);
    if (candidate[1] >= 0.0)
      {
        const double x = x_upper
      }
  }
  double BndNaca4Digit::solve_parameter (const Point<2> &candidate) const
  {
    const double x = candidate[0]/chord_length;
    const double y = candidate[1]/chord_length;
    if (x<0.001)
      {
        Assert (x>=-0.01, ExcMessage ("Point not on airfoil (LE)"));
        // Solve Y equation for parameter
        if (y > 2.e-6)
          {
            Assert (y<= max_thickness + max_camber + 0.005,
                    ExcMessage ("Point not on airfoil (LE Upper)"));

            double x_l = 0.0;
            double y_l = 0.0;
            double x_r = x_try;
            double y_r = -1.0;

            {
              Fad_db x_r_ad = x_r;
              x_r_ad.diff (0,1);
              y_r = y_upper (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
            }

            if (y_r == y)
              {
                return (x);
              }

            while (y_r < y)
              {
                x_l = x_r;
                y_l = y_r;
                x_r = 2.0 * x_l;
                Fad_db x_r_ad = x_r;
                x_r_ad.diff (0,1);
                y_r = y_upper (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
              }
            if (y_r == y)
              {
                return (x);
              }

            while (std::abs (x_r-x_l) >= 1.0e-10)
              {
                const double x_try = (y-yl)* (x_r-x_l)/ (y_r-yl);
                Fad_db x_r_ad = x_try;
                x_r_ad.diff (0,1);
                y_try = y_upper (x_try, std::atan (camber (x_r_ad).fastAccessDx (0)));

                // These two if statements also handle (y_try == y)
                if (y_try >= y)
                  {
                    x_r = x_try;
                  }
                if (y_try <= y)
                  {
                    x_l = x_try;
                  }
              }
            return (0.5* (x_l+x_r));
          }
        else if (y < -2.e-6)
          {
            Assert (y>= -max_thickness + max_camber - 0.005,
                    ExcMessage ("Point not on airfoil (LE Lower)"));
            double x_l = 0.0;
            double y_l = 0.0;
            double x_r = x_try;
            double y_r = 1.0;

            {
              Fad_db x_r_ad = x_r;
              x_r_ad.diff (0,1);
              y_r = y_lower (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
            }

            if (y_r == y)
              {
                return (x);
              }

            while (y_r > y)
              {
                x_l = x_r;
                y_l = y_r;
                x_r = 2.0 * x_l;
                Fad_db x_r_ad = x_r;
                x_r_ad.diff (0,1);
                y_r = y_lower (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
              }
            if (y_r == y)
              {
                return (x);
              }

            while (std::abs (x_r-x_l) >= 1.0e-10)
              {
                const double x_try = (y-yl)* (x_r-x_l)/ (y_r-yl);
                Fad_db x_r_ad = x_try;
                x_r_ad.diff (0,1);
                y_try = y_lower (x_try, std::atan (camber (x_r_ad).fastAccessDx (0)));

                // These two if statements also handle (y_try == y)
                if (y_try <= y)
                  {
                    x_r = x_try;
                  }
                if (y_try >= y)
                  {
                    x_l = x_try;
                  }
              }
            return (0.5* (x_l+x_r));
          }
        else
          {
            return (0.0);
          }
      }
    else
      {
        // Solve X equation for parameter
        Assert (x<=1.0, ExcMessage ("Point not on airfoil (TE)"));
        if (y > 0.0)
          {
            Assert (y <= max_thickness + max_camber + 0.005,
                    ExcMessage ("Point not on airfoil (TE Upper)"));
            double return_value = x;
            while (true)
              {
                Fad_db x_ad = return_value;
                x_ad.diff (0,1);
                FFad_db x_ad_ad = x_ad;
                x_ad_ad.diff (0,1);
                Fad_db res_ad = x_upper (x_r_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));

                if (std::abs (res_ad.val()) < 1.0e-10)
                  {
                    break;
                  }
                return_value -= res_ad.fastAccessDx (0);
              };
          }
        return (return_value);
      }
    else if (y < 0.0)
      {
        Assert (y>= -max_thickness + max_camber - 0.005,
                ExcMessage ("Point not on airfoil (TE Lower)"));
        if (y > 0.0)
          {
            Assert (y <= max_thickness + max_camber + 0.005,
                    ExcMessage ("Point not on airfoil (TE Upper)"));
            double return_value = x;
            while (true)
              {
                Fad_db x_ad = return_value;
                x_ad.diff (0,1);
                FFad_db x_ad_ad = x_ad;
                x_ad_ad.diff (0,1);
                Fad_db res_ad = x_lower (x_r_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));

                if (std::abs (res_ad.val()) < 1.0e-10)
                  {
                    break;
                  }
                return_value -= res_ad.fastAccessDx (0);
              };
          }
        return (return_value);
      }
    else
      {
        return (1.0);
      }
  }
}

}
