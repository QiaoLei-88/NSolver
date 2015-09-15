//
//  BndNaca4DigitSymm.h
//  NSolver
//
//  Created by 乔磊 on 15/9/14.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#ifndef __BndNaca4DigitSymm__H__
#define __BndNaca4DigitSymm__H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_boundary_lib.h>

// On OS X  system, you need to include this header;
// On Linux, you don't.
#include <deal.II/lac/trilinos_solver.h>


// Sacado is the automatic differentiation package within Trilinos, which is
// used to find the Jacobian for a fully implicit Newton iteration:
// Trilinos::Sacado (at least until version 11.10.2) package will trigger
// warnings when compiling this file. Since we are not responsible for this,
// we just suppress the warning by wrapping the <code>#include</code>
// directive into a pair of macros that simply suppress these warnings:
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <Sacado.hpp>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS


namespace NSFEMSolver
{
  using namespace dealii;

  class BndNaca4DigitSymm: public StraightBoundary<2>
  {
  public:
    BndNaca4DigitSymm (const unsigned int number,
                       const double chord_length_in);
    ~BndNaca4DigitSymm();
    /**
     * Let the new point be the arithmetic mean of the two vertices of the line.
     *
     * Refer to the general documentation of this class and the documentation of
     * the base class for more information.
     */
    virtual
    Point<2>
    get_new_point_on_line (const typename Triangulation<2,2>::line_iterator &line) const;

    // /**
    //  * Gives <tt>n=points.size()</tt> points that splits the StraightBoundary
    //  * line into $n+1$ partitions of equal lengths.
    //  *
    //  * Refer to the general documentation of this class and the documentation of
    //  * the base class.
    //  */
    // virtual
    // void
    // get_intermediate_points_on_line (const typename Triangulation<2,2>::line_iterator &line,
    //                                  std::vector<Point<2> > &points) const;

    /**
      * Implementation of the function declared in the base class.
      *
      * Refer to the general documentation of this class and the documentation of
      * the base class.
      */
    // virtual
    // Tensor<1,2>
    // normal_vector (const typename Triangulation<2,2>::face_iterator &face,
    //                const Point<2> &p) const;

    // /**
    //  * Compute the normals to the boundary at the vertices of the given face.
    //  *
    //  * Refer to the general documentation of this class and the documentation of
    //  * the base class.
    //  */
    // virtual
    // void
    // get_normals_at_vertices (const typename Triangulation<2,2>::face_iterator &face,
    //                          typename Boundary<2,2>::FaceVertexNormals &face_vertex_normals) const;

    /**
     * Given a candidate point and a line segment characterized by the iterator,
     * return a point that lies on the surface described by this object. This
     * function is used in some mesh smoothing algorithms that try to move
     * around points in order to improve the mesh quality but need to ensure
     * that points that were on the boundary remain on the boundary.
     *
     * The point returned is the projection of the candidate point onto the line
     * through the two vertices of the given line iterator.
     *
     * If 2==1, then the line represented by the line iterator is the
     * entire space (i.e. it is a cell, not a part of the boundary), and the
     * returned point equals the given input point.
     */
    // virtual
    // Point<2>
    // project_to_surface (const typename Triangulation<2,2>::line_iterator &line,
    //                     const Point<2> &trial_point) const;

    /**
     * Test driver for private functions.
     */
    void test() const;

  private:
    typedef Sacado::Fad::DFad<double> Fad_db;
    typedef Sacado::Fad::DFad<Fad_db> FFad_db;
    double solve_mid_curve (const Point<2> &candidate) const;
    template<typename Number>
    Number thickness (const Number &x) const;

    template<typename Number>
    Number fitted_curve_length (const Number &x) const;

    double max_thickness;
    double chord_length;
  };


  template<typename Number>
  inline
  Number BndNaca4DigitSymm::thickness (const Number &x) const
  {
    Assert (x>=0.0, ExcMessage ("Point not on airfoil camber"));
    Assert (x<=1.0, ExcMessage ("Point not on airfoil camber"));
    return (5.0* max_thickness * (0.2969 * std::sqrt (x)
                                  + (-0.1260 + (-0.3516 + (0.2843 - 0.1036*x) *x) *x) *x
                                 ));
  }

  template<typename Number>
  Number BndNaca4DigitSymm::fitted_curve_length (const Number &x) const
  {
    // This function is hard coded according NACA0012 foil formula.
    if (x<0.059)
      {
        return (2.32049366940377*x*x+0.155330075545698*std::sqrt (x+1.0e-16)+0.482354339318731*x);
      }
    else
      {
        return (0.01526608872+x);
      }
  }
}
#endif
