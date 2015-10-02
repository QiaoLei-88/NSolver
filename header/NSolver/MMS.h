//
//  MMS.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__MMS__
#define __NSolver__MMS__

#include <NSolver/NSEquation.h>

namespace NSFEMSolver
{
  using namespace dealii;

// MMS
  struct Coeff_2D
  {
    /**
     * Default constructor
     */
    Coeff_2D();

    /**
     * Copy constructor
     */
    Coeff_2D (const Coeff_2D &value_in);

    /**
     * Overload assign operator
     */
    Coeff_2D &operator= (const Coeff_2D &r);

    /**
     * Value list
     */
    double c0;
    double cx;
    double cy;
    double cxy;
    double ax;
    double ay;
    double axy;
  };

#define dim2 2
  /**
   * Class for carrying out code verification by the method of manufactured
   * solutions (MMS). For now, the test function is hard coded and only the
   * coefficients can be adjusted.
   *
   * By default, this class works in Euler mode, i.e., no viscous flux
   * will be involved.
   */
  class MMS
  {
  public:
    typedef Sacado::Fad::DFad<double> FADD;
    // Fluid vector
    typedef std_cxx11::array<double, EquationComponents<dim2>::n_components>  F_V;
    // Fluid tensor
    typedef std_cxx11::array <std_cxx11::array <double, dim2>,
            EquationComponents<dim2>::n_components > F_T;

    typedef std_cxx11::array<FADD, EquationComponents<dim2>::n_components> FADD_V;
    typedef std_cxx11::array <std_cxx11::array <FADD, dim2>,
            EquationComponents<dim2>::n_components > FADD_T;
    // coefficient vector
    typedef std_cxx11::array<Coeff_2D, EquationComponents<dim2>::n_components> C_V;
  public:
    /**
     * Default constructor
     */
    MMS();


    /**
     * Copy constructor
     */
    MMS (const MMS &mms_in);


    /**
     * Initialize the object with provided coefficients. This has nothing to
     * do with whether the calling object will work in Euler or Navier-Stokes
     * mode.
     */
    void reinit
    (C_V &c_in);


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
                   F_V &value,
                   F_V &source,
                   const bool need_source = false) const;
  private:
    C_V c;
    bool initialized;
    bool is_NS;
  };
}
#endif
