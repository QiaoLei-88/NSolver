//
//  MMS.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__MMS__
#define __NSolver__MMS__

#include <deal.II/base/function.h>

#include <NSolver/NSEquation.h>

namespace NSFEMSolver
{
  using namespace dealii;

  // Code for manufactured solutions

  struct Coeff
  {
    /**
     * Default constructor
     */
    Coeff();

    /**
     * Copy constructor
     */
    Coeff(const Coeff &value_in);

    /**
     * Overload assign operator
     */
    Coeff &
    operator=(const Coeff &r);

    /**
     * Value list
     */
    double c0;
    double cx;
    double cy;
    double cz;
    double cxy;
    double cxz;
    double cxyz;
    double ax;
    double ay;
    double az;
    double axy;
    double axz;
    double axyz;
  };


  /**
   * Class for carrying out code verification by the method of manufactured
   * solutions (MMS). For now, the test function is hard coded and only the
   * coefficients can be adjusted.
   *
   * By default, this class works in Euler mode, i.e., no viscous flux
   * will be involved.
   */
  template <int dim>
  class MMS : public Function<dim, double>
  {
  public:
    typedef Sacado::Fad::DFad<double> FADD;
    // Fluid vector
    typedef std::array<double, EquationComponents<dim>::n_components> F_V;
    // Fluid tensor
    typedef std::array<std::array<double, dim>,
                       EquationComponents<dim>::n_components>
      F_T;

    typedef std::array<FADD, EquationComponents<dim>::n_components> FADD_V;
    typedef std::array<std::array<FADD, dim>,
                       EquationComponents<dim>::n_components>
      FADD_T;
    // coefficient vector
    typedef std::array<Coeff, EquationComponents<dim>::n_components> C_V;

  public:
    /**
     * Default constructor
     */
    MMS();

    /**
     * Copy constructor
     */
    MMS(const MMS<dim> &mms_in);

    /**
     * Initialize the object with provided coefficients. This has nothing to
     * do with whether the calling object will work in Euler or Navier-Stokes
     * mode.
     */
    void
    reinit(C_V &c_in);

    /**
     * You can control the equation mode of the MMS object by the flowing two
     * member functions.
     */
    void
    set_eqn_to_NS();
    void
    set_eqn_to_Euler();

    /**
     * return the manufactured solution is subsonic or not.
     */
    bool
    is_subsonic() const;

    /**
     * Evaluate the exact solution at specified point p and put result into
     * array value. If need_source is set to true, the corresponding source
     * term at point p will be evaluated and put into array source.
     *
     * The source term is the divergence of the flux of the governing equation.
     * The divergence is evaluated by automatic differentiation with
     * Sacado::FAD.
     */
    void
    evaluate(const Point<dim> &p,
             F_V &             value,
             F_T &             grad,
             F_V &             source,
             const bool        need_source = false) const;
    /**
     * access to one component at one point
     */
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;
    /**
     * return all components at one point
     */
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const;

    /**
     * access to one component at several points
     */
    virtual void
    value_list(const std::vector<Point<dim>> &point_list,
               std::vector<double> &          value_list,
               const unsigned int             component = 0) const;
    /**
     * return all components at several points
     */
    virtual void
    vector_value_list(const std::vector<Point<dim>> &point_list,
                      std::vector<Vector<double>> &  value_list) const;

  private:
    C_V  c;
    bool initialized;
    bool is_NS;

    template <typename Number>
    void
    value_at_point(const std::array<Number, dim> &p,
                   const unsigned int             component,
                   Number &                       result) const;
  };
} // namespace NSFEMSolver
#endif
