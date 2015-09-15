//
//  BndNaca4DigitSymm.h
//  NSolver
//
//  Created by 乔磊 on 15/9/14.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#include <NSolver/BoundaryManifold/BndNaca4DigitSymm.h>

namespace NSFEMSolver
{
  using namespace dealii;

  BndNaca4DigitSymm::BndNaca4DigitSymm (const unsigned int /*number*/,
                                        const double chord_length_in)
    :
    // max_thickness (static_cast<double> (number%100)/100.0),
    max_thickness (0.12),
    chord_length (chord_length_in)
  {}

  BndNaca4DigitSymm::~BndNaca4DigitSymm()
  {}


  Point<2>
  BndNaca4DigitSymm::get_new_point_on_line (const typename Triangulation<2,2>::line_iterator &line) const
  {

#ifdef DEBUG_BndNaca4DigitSymm
    std::cerr << "\n\n\n";
    static std::ofstream new_points_out ("new_points_debug.txt");
    static std::ofstream old_points_out ("old_points_debug.txt");
    static std::ofstream solve_out ("solve_test.txt");

    new_points_out << std::scientific;
    new_points_out.precision (8);
    old_points_out << std::scientific;
    old_points_out.precision (8);
    solve_out << std::scientific;
    solve_out.precision (8);

    old_points_out << line->vertex (0)[0] << "\t" << line->vertex (0)[1] << std::endl;
    old_points_out << line->vertex (1)[0] << "\t" << line->vertex (1)[1] << std::endl;
#endif

    const Point<2> &p1 = line->vertex (0);
    const Point<2> &p2 = line->vertex (1);
    const Point<2> candidate = (p1 + p2) / 2.0;

#ifdef DEBUG_BndNaca4DigitSymm
    const unsigned int bc_id = line->boundary_id();
    std::cerr << bc_id << ": "
              << line->vertex (0) << "; "
              << line->vertex (1) << std::endl;
    std::cerr << "candidate : " << candidate << std::endl;
#endif
    if (line->boundary_id() == 0)
      {
        return (candidate);
      }

    double s1;
    double s2;
    // Curve length from LE to the two points
    if (std::abs (p1[1]) < 1e-26)
      {
        s1 = fitted_curve_length (p1[0]) * (1.0 - 2.0*std::signbit (p2[1]));
      }
    else
      {
        s1 = fitted_curve_length (p1[0]) * (1.0 - 2.0*std::signbit (p1[1]));
      }
    if (std::abs (p2[1]) < 1e-26)
      {
        s2 = fitted_curve_length (p2[0]) * (1.0 - 2.0*std::signbit (p1[1]));
      }
    else
      {
        s2 = fitted_curve_length (p2[0]) * (1.0 - 2.0*std::signbit (p2[1]));
      }
    //std::abs(p1[1]) and std::abs(p2[1]) should not approximate to zero at same time,
    // which means p1 and p2 are trailing and leading edge.

    // Solve for position with medium curve length with Newton method
    const double s_mid_abs = std::abs (0.5* (s1+s2));
    const double s_mid_sign = 1.0 - 2.0*std::signbit (s1+s2);

    double sol = std::max (candidate[0], 1.0e-16);

    //Newton iteration
#ifdef DEBUG_BndNaca4DigitSymm
    std::cerr << "\n solve mid curve" << std::endl;
#endif
    for (unsigned int n_iter = 0; n_iter < 100; ++n_iter)
      {
        Fad_db x_ad = sol;
        x_ad.diff (0,1);

        Fad_db res_ad = fitted_curve_length<Fad_db> (x_ad) - s_mid_abs;
#ifdef DEBUG_BndNaca4DigitSymm
        std::cerr << res_ad.val() << ",   " << sol << std::endl;
#endif
        if (std::abs (res_ad.val()) < 1.0e-10)
          {
            break;
          }
        sol -= res_ad.val()/res_ad.fastAccessDx (0);
        sol = std::max (sol, 1.0e-16);
#ifdef DEBUG_BndNaca4DigitSymm
        std::cerr << sol << std::endl;
#endif
      }
    const double y_foil = thickness (sol) * s_mid_sign;
#ifdef DEBUG_BndNaca4DigitSymm
    solve_out << candidate[0] << "  " << sol << "  " << candidate[0] - sol << "  "
              << candidate[1] << "  " << y_foil << "  " << candidate[1] - y_foil << "  "
              << std::endl;
#endif
    return (Point<2> (sol, y_foil));
  }


  // Tensor<1,2>
  // BndNaca4DigitSymm::normal_vector (const typename Triangulation<2,2>::face_iterator &face,
  //                                   const Point<2> &p) const
  // {
  //   std::cerr << "\n normal_vector : "
  //             << p << std::endl;

  //   const double x = solve_parameter (p);

  //   Fad_db x_ad = x;
  //   x_ad.diff (0,1);
  //   FFad_db x_ad_ad = x_ad;
  //   x_ad_ad.diff (0,1);
  //   Tensor<1,2> return_value;

  //   if (p[1] >= 0.0)
  //     {
  //       Fad_db x_foil = x_upper<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
  //       Fad_db y_foil = y_upper<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
  //       return_value[0] = -y_foil.fastAccessDx (0);
  //       return_value[1] =  x_foil.fastAccessDx (0);
  //     }
  //   else
  //     {
  //       Fad_db x_foil = x_lower<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
  //       Fad_db y_foil = y_lower<Fad_db> (x_ad, std::atan (camber (x_ad_ad).fastAccessDx (0)));
  //       return_value[0] =  y_foil.fastAccessDx (0);
  //       return_value[1] = -x_foil.fastAccessDx (0);
  //     }
  //   return_value /= return_value.norm();
  //   return (return_value);
  // }


  // Point<2>
  // BndNaca4DigitSymm::project_to_surface (const typename Triangulation<2,2>::line_iterator &line,
  //                                        const Point<2> &trial_point) const
  // {
  //   std::cerr << "\n project_to_surface : "
  //             << trial_point << std::endl;

  //   const Point<2> &p1 = line->vertex (0);
  //   const Point<2> &p2 = line->vertex (1);

  //   const double s = (trial_point-p1)* (p2-p1) / ((p2-p1)* (p2-p1));

  //   Assert (s > -0.5,
  //           ExcMessage ("Project source point is to far in negative."));
  //   Assert (s < 1.5,
  //           ExcMessage ("Project source point is to far in positive."));

  //   const Point<2> candidate = p1 + s* (p2-p1);

  //   const double x = solve_parameter (candidate);
  //   Fad_db x_ad = x;
  //   x_ad.diff (0,1);

  //   if (candidate[1] >= 0.0)
  //     {
  //       double x_foil = x_upper (x, std::atan (camber (x_ad).fastAccessDx (0)));
  //       double y_foil = y_upper (x, std::atan (camber (x_ad).fastAccessDx (0)));
  //       return (Point<2> (x_foil, y_foil));
  //     }
  //   else
  //     {
  //       double x_foil = x_lower (x, std::atan (camber (x_ad).fastAccessDx (0)));
  //       double y_foil = y_lower (x, std::atan (camber (x_ad).fastAccessDx (0)));
  //       return (Point<2> (x_foil, y_foil));
  //     }
  // }
}
