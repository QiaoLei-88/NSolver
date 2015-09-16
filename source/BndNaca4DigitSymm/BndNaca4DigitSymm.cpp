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


  Tensor<1,2>
  BndNaca4DigitSymm::normal_vector (const typename Triangulation<2,2>::face_iterator &face,
                                    const Point<2> &p) const
  {
    const double x = std::max (p[0], 1e-16);
    Assert (p[0] > -1e-6, ExcMessage ("Point not on foil"));
    Assert (p[0] <  1.0 + 1e-6, ExcMessage ("Point not on foil"));

    Fad_db x_ad = x;
    x_ad.diff (0,1);

    Fad_db y = thickness<Fad_db> (x_ad);
    Assert (std::abs (p[1] - y) < 1e-6, ExcMessage ("Point not on foil"));
    Tensor<1,2> return_value;
    return_value[0] = y.fastAccessDx (0);
    if (p[1] > 0.0)
      {
        return_value[1] = -1.0;
      }
    else if (p[1] < 0.0)
      {
        return_value[1] = 1.0;
      }
    else //if (p[1]  == 0.0)
      {
        bool norm_setted = false;
        for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_face; ++v)
          {
            if (face->vertex (v)[1] != 0.0)
              {
                if (face->vertex (v)[1] > 0.0)
                  {
                    return_value[1] = -1.0;
                  }
                else // if (face->vertex (v)[1]] < 0.0)
                  {
                    return_value[1] = 1.0;
                  }
                norm_setted = true;
              }
            Assert (norm_setted, ExcMessage ("All vertices on face have zero Y"));
          }
      }
    return_value /= return_value.norm();
    return (return_value);
  }

  void
  BndNaca4DigitSymm::get_normals_at_vertices (const typename Triangulation<2,2>::face_iterator &face,
                                              typename Boundary<2,2>::FaceVertexNormals &face_vertex_normals) const
  {
    // Beware upper_or_lower == 1.0 for lower half of the foil
    // and upper_or_lower == -1.0 for upper half of the foil.
    // Because we want the outer normal vector of the flow field
    // rather than the air foil.
    // This may be a surprise.
    double upper_or_lower = 0.0;
    for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_face; ++v)
      {
        if (face->vertex (v)[1] > 0.0)
          {
            upper_or_lower = -1.0;
          }
        else
          {
            upper_or_lower = 1.0;
          }
      }
    Assert (upper_or_lower != 0.0, ExcMessage ("All vertices on face have zero Y"));

    for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_face; ++v)
      {
        const Point<2> &p = face->vertex (v);
        const double x = std::max (p[0], 1e-16);
        Assert (p[0] > -1e-6, ExcMessage ("Point not on foil"));
        Assert (p[0] <  1.0 + 1e-6, ExcMessage ("Point not on foil"));

        Fad_db x_ad = x;
        x_ad.diff (0,1);

        Fad_db y = thickness<Fad_db> (x_ad);
        Assert (std::abs (p[1] - y) < 1e-6, ExcMessage ("Point not on foil"));
        Tensor<1,2> return_value;
        return_value[0] = y.fastAccessDx (0);
        return_value[1] = upper_or_lower;
        face_vertex_normals[v] = return_value/return_value.norm();
      }
    return;
  }


}
