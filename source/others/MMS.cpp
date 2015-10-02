//
//  MMS.cpp
//  NSolver
//
//  Created by 乔磊 on 15/4/24.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/NSEquation.h>
#include <NSolver/MMS.h>

namespace NSFEMSolver
{
  Coeff_2D::Coeff_2D()
  {}

  Coeff_2D::Coeff_2D (const Coeff_2D &value_in)
    :
    c0 (value_in.c0),
    cx (value_in.cx),
    cy (value_in.cy),
    cxy (value_in.cxy),
    ax (value_in.ax),
    ay (value_in.ay),
    axy (value_in.axy)
  {}

  Coeff_2D &Coeff_2D::operator= (const Coeff_2D &r)
  {
    this -> c0  = r.c0 ;
    this -> cx  = r.cx ;
    this -> cy  = r.cy ;
    this -> cxy = r.cxy;
    this -> ax  = r.ax ;
    this -> ay  = r.ay ;
    this -> axy = r.axy;
    return (*this);
  }

  MMS::MMS()
    :
    Function<dim2> (EquationComponents<dim2>::n_components),
    initialized (false),
    is_NS (false)
  {}


  MMS::MMS (const MMS &mms_in)
    :
    Function<dim2> (EquationComponents<dim2>::n_components),
    initialized (mms_in.initialized),
    is_NS (mms_in.is_NS)
    // TODO: It seems now the deal.II provided std_cxx11::array does not
    // support copying initialization, so do it by 'for loop'.
  {
    for (unsigned int i=0; i<EquationComponents<dim2>::n_components; ++i)
      {
        c[i] = mms_in.c[i];
      }
  }

  void MMS::reinit
  (std_cxx11::array<Coeff_2D, EquationComponents<dim2>::n_components> &c_in)
  {
    for (unsigned int ic=0; ic<EquationComponents<dim2>::n_components; ++ic)
      {
        c[ic] = c_in[ic];
      }
    initialized = true;
    return;
  }

  void MMS::set_eqn_to_NS()
  {
    is_NS = true;
    return;
  }
  void MMS::set_eqn_to_Euler()
  {
    is_NS = false;
    return;
  }

  bool MMS::is_subsonic() const
  {
    // This implementation id no reliable, but works under current situation.
    // Mean velocity square
    return ((c[0].c0*c[0].c0 + c[1].c0*c[1].c0) < 1.0);
  }

  double MMS::value (const Point<dim2>  &p,
                     const unsigned int component) const
  {
    double rv (0.0);
    value_at_point<double> (p[0], p[1], component, rv);
    return (rv);
  }

  void MMS::vector_value (const Point<dim2> &p,
                          Vector<double>   &value) const
  {
    Assert (value.size() >= this->n_components,
            ExcMessage ("Not enough space for all components!"));
    for (unsigned int i=0; i<this->n_components; ++i)
      {
        value[i] = this->value (p,i);
      }
    return;
  }

  void MMS::value_list (const std::vector<Point<dim2> > &point_list,
                        std::vector<double>             &value_list,
                        const unsigned int  component) const
  {
    Assert (point_list.size() == value_list.size(),
            ExcMessage ("Vector size mismatch!"));
    for (unsigned int i=0; i<point_list.size(); ++i)
      {
        value_list[i] = this->value (point_list[i], component);
      }
    return;
  }

  void MMS::vector_value_list (const std::vector<Point<dim2> > &point_list,
                               std::vector<Vector<double> >    &value_list) const
  {
    Assert (point_list.size() == value_list.size(),
            ExcMessage ("Vector size mismatch!"));
    for (unsigned int i=0; i<point_list.size(); ++i)
      {
        this->vector_value (point_list[i], value_list[i]);
      }
    return;
  }

  void MMS::evaluate (const Point<dim2>   &p,
                      F_V &value,
                      F_T &grad,
                      F_V &source,
                      const bool need_source) const
  {
    Assert (initialized, ExcMessage ("run MMS::reinit(...) before MMS::evaluation(...)."));

    FADD x=p (0);
    FADD y=p (1);
    x.diff (0,2);
    y.diff (1,2);
    FADD_V value_ad;
    for (unsigned int ic=0; ic<EquationComponents<dim2>::n_components; ++ic)
      {
        value_ad[ic] = 0.0;
      }

    for (unsigned int ic=0; ic<EquationComponents<dim2>::n_components; ++ic)
      {
        value_at_point<FADD> (x,y,ic,value_ad[ic]);
      }

    for (unsigned int ic=0; ic<EquationComponents<dim2>::n_components; ++ic)
      {
        value[ic] = value_ad[ic].val();
        for (unsigned int d=0; d<dim2; ++d)
          {
            grad[ic][d] = value_ad[ic].dx (d);
          }
      }

    if (need_source)
      {
        {
          FADD_T flux;

          EulerEquations<dim2>::compute_inviscid_flux (value_ad, flux);

          for (unsigned int ic=0; ic<EquationComponents<dim2>::n_components; ++ic)
            {
              source[ic] = flux[ic][0].dx (0) + flux[ic][1].dx (1);
            }
        }
        if (is_NS)
          {
            FADD_T grad_value;
            for (unsigned int ic=0; ic<EquationComponents<dim2>::n_components; ++ic)
              {
                grad_value[ic][0] = value_ad[ic].dx (0);
                grad_value[ic][1] = value_ad[ic].dx (1);
              }

            FADD_T visc_flux;

            EulerEquations<dim2>::compute_viscous_flux (value_ad, grad_value, visc_flux);

            for (unsigned int ic=0; ic<EquationComponents<dim2>::n_components; ++ic)
              {
                source[ic] -= (visc_flux[ic][0].dx (0) + visc_flux[ic][1].dx (1));
              }
          }
      }
    return;
  }

  template<typename Number>
  void MMS::value_at_point (const Number &x,
                            const Number &y,
                            const unsigned int component,
                            Number &result) const
  {
    const double &Pi = numbers::PI;
    const Coeff_2D &t = c[component];
    switch (component)
      {
      case EquationComponents<dim2>::first_momentum_component:
        result += t.c0;
        result += t.cx  * std::sin (t.ax  * Pi * x);
        result += t.cy  * std::cos (t.ay  * Pi * y);
        result += t.cxy * std::cos (t.axy * Pi * x * y);
        break;
      case EquationComponents<dim2>::first_momentum_component + 1:
        result += t.c0;
        result += t.cx  * std::cos (t.ax  * Pi * x);
        result += t.cy  * std::sin (t.ay  * Pi * y);
        result += t.cxy * std::cos (t.axy * Pi * x * y);
        break;
      case EquationComponents<dim2>::density_component:
        result += t.c0;
        result += t.cx  * std::sin (t.ax  * Pi * x);
        result += t.cy  * std::cos (t.ay  * Pi * y);
        result += t.cxy * std::cos (t.axy * Pi * x * y);
        break;
      case EquationComponents<dim2>::pressure_component:
        result += t.c0;
        result += t.cx  * std::cos (t.ax  * Pi * x);
        result += t.cy  * std::sin (t.ay  * Pi * y);
        result += t.cxy * std::sin (t.axy * Pi * x * y);
        break;
      default:
        Assert (false, ExcNotImplemented());
        break;
      }
    return;
  }
}
