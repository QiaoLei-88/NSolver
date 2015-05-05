//
//  MMS.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__MMS__
#define __NSolver__MMS__

#include "NSEquation.h"

namespace NSolver
{
  using namespace dealii;

// MMS
  struct Coeff_2D
  {
    double c0;
    double cx;
    double cy;
    double cxy;
    double ax;
    double ay;
    double axy;
    Coeff_2D &operator= (const Coeff_2D &r);
  };

#define dim2 2
  /**
   * Class for carrying out code verification by the method of manufactured
   * solutions (MMS). For now, the test function is hard coded and only the
   * coefficients can be adjustd.
   *
   * By default, this class works in Euler mode, i.e., no viscous flux
   * will be involved.
   */
  class MMS
  {
  public:
    MMS();
    /**
     * Initialize the object with provided coefficients. This has nothing to
     * do with whether the calling object will work in Euler or Navier-Stokes
     * mode.
     */
    void reinit
    (std_cxx11::array<Coeff_2D, EulerEquations<dim2>::n_components> &c_in);


    /**
     * You can control the equation mode of the MMS object by the flowing two
     * member functions.
     */
    void set_eqn_to_NS();
    void set_eqn_to_Euler();

    /**
     * Evaluate the exact solution at specified point p and put result into
     * array value. If need_source is set to true, the corresponding source
     * term at point p will be evaluated and put into array source.
     *
     * The source term is the divergence of the flux of the governing equation.
     * The divergence is evaluated by automatic differentiation with Sacado::FAD.
     */
    void evaluate (const Point<dim2>   &p,
                   std_cxx11::array<double, EulerEquations<dim2>::n_components> &value,
                   std_cxx11::array<double, EulerEquations<dim2>::n_components> &source,
                   const bool need_source = false) const;
  private:
    std_cxx11::array<Coeff_2D, EulerEquations<dim2>::n_components> c;
    bool initialized;
    bool is_NS;
  };
}
#endif
