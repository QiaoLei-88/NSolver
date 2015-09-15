


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

  BndNaca4Digit::~BndNaca4Digit()
  {}


  Point<2>
  BndNaca4Digit::get_new_point_on_line (const typename Triangulation<2,2>::line_iterator &line) const
  {
    const Point<2> candidate = (line->vertex (0) + line->vertex (1)) / 2.0;
    const double x = solve_parameter (candidate);
    Fad_db x_ad = x;
    x_ad.diff (0,1);

    if (candidate[1] >= 0.0)
      {
        double x_foil = x_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        double y_foil = y_upper<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        return (Point<2> (x_foil, y_foil));
      }
    else
      {
        double x_foil = x_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        double y_foil = y_lower<double> (x, std::atan (camber (x_ad).fastAccessDx (0)));
        return (Point<2> (x_foil, y_foil));
      }
  }


  Tensor<1,2>
  BndNaca4Digit::normal_vector (const typename Triangulation<2,2>::face_iterator &face,
                                const Point<2> &p) const
  {
    const double x = solve_parameter (p);

    Fad_db x_ad = x;
    x_ad.diff (0,1);
    FFad_db x_ad_ad = x_ad;
    x_ad_ad.diff (0,1);
    Tensor<1,2> return_value;

    if (p[1] >= 0.0)
      {
        Fad_db x_foil = x_upper<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
        Fad_db y_foil = y_upper<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
        return_value[0] =  y_foil.fastAccessDx (0);
        return_value[1] = -x_foil.fastAccessDx (0);
      }
    else
      {
        Fad_db x_foil = x_lower<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
        Fad_db y_foil = y_lower<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
        return_value[0] = -y_foil.fastAccessDx (0);
        return_value[1] =  x_foil.fastAccessDx (0);
      }
    return_value /= return_value.norm();
    return (return_value);
  }


  Point<2>
  BndNaca4Digit::project_to_surface (const typename Triangulation<2,2>::line_iterator &line,
                                     const Point<2> &trial_point) const
  {
    const Point<2> &p1 = line->vertex (0);
    const Point<2> &p2 = line->vertex (1);

    const double s = (trial_point-p1)* (p2-p1) / ((p2-p1)* (p2-p1));

    Assert (s > -0.5,
            ExcMessage ("Project source point is to far in negative."));
    Assert (s < 1.5,
            ExcMessage ("Project source point is to far in positive."));

    const Point<2> candidate = p1 + s* (p2-p1);

    const double x = solve_parameter (candidate);
    Fad_db x_ad = x;
    x_ad.diff (0,1);

    if (candidate[1] >= 0.0)
      {
        double x_foil = x_upper (x, std::atan (camber (x_ad).fastAccessDx (0)));
        double y_foil = y_upper (x, std::atan (camber (x_ad).fastAccessDx (0)));
        return (Point<2> (x_foil, y_foil));
      }
    else
      {
        double x_foil = x_lower (x, std::atan (camber (x_ad).fastAccessDx (0)));
        double y_foil = y_lower (x, std::atan (camber (x_ad).fastAccessDx (0)));
        return (Point<2> (x_foil, y_foil));
      }
  }


  double
  BndNaca4Digit::solve_parameter (const Point<2> &candidate) const
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
            double x_r = x;
            double y_r = -1.0;

            {
              Fad_db x_r_ad = x_r;
              x_r_ad.diff (0,1);
              y_r = y_upper<double> (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
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
                y_r = y_upper<double> (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
              }
            if (y_r == y)
              {
                return (x);
              }

            while (std::abs (x_r-x_l) >= 1.0e-10)
              {
                const double x_try = (y-y_l)* (x_r-x_l)/ (y_r-y_l);
                Fad_db x_r_ad = x_try;
                x_r_ad.diff (0,1);
                const double y_try =
                  y_upper<double> (x_try, std::atan (camber (x_r_ad).fastAccessDx (0)));

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
            double x_r = x;
            double y_r = 1.0;

            {
              Fad_db x_r_ad = x_r;
              x_r_ad.diff (0,1);
              y_r = y_lower<double> (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
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
                y_r = y_lower<double> (x_r, std::atan (camber (x_r_ad).fastAccessDx (0)));
              }
            if (y_r == y)
              {
                return (x);
              }

            while (std::abs (x_r-x_l) >= 1.0e-10)
              {
                const double x_try = (y-y_l)* (x_r-x_l)/ (y_r-y_l);
                Fad_db x_r_ad = x_try;
                x_r_ad.diff (0,1);
                const double y_try =
                  y_lower<double> (x_try, std::atan (camber (x_r_ad).fastAccessDx (0)));

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
    else //if (x<0.001)
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
                Fad_db res_ad = x_upper<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));

                if (std::abs (res_ad.val()) < 1.0e-10)
                  {
                    break;
                  }
                return_value -= res_ad.fastAccessDx (0);
              };
            return (return_value);
          }
        else if (y < 0.0)
          {
            Assert (y>= -max_thickness + max_camber - 0.005,
                    ExcMessage ("Point not on airfoil (TE Lower)"));

            double return_value = x;
            while (true)
              {
                Fad_db x_ad = return_value;
                x_ad.diff (0,1);
                FFad_db x_ad_ad = x_ad;
                x_ad_ad.diff (0,1);
                Fad_db res_ad = x_lower<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));

                if (std::abs (res_ad.val()) < 1.0e-10)
                  {
                    break;
                  }
                return_value -= res_ad.fastAccessDx (0);
              };
            return (return_value);
          }
        else
          {
            return (1.0);
          }
      } // if (x<0.001)
  } // End function BndNaca4Digit::solve_parameter

}
