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
  Coeff::Coeff()
    :
    c0 (0.0),
    cx (0.0),
    cy (0.0),
    cz (0.0),
    cxy (0.0),
    cxz (0.0),
    cxyz (0.0),
    ax (0.0),
    ay (0.0),
    az (0.0),
    axy (0.0),
    axz (0.0),
    axyz (0.0)
  {}

  Coeff::Coeff (const Coeff &value_in)
    :
    c0 (value_in.c0)    ,
    cx (value_in.cx)    ,
    cy (value_in.cy)    ,
    cz (value_in.cz)    ,
    cxy (value_in.cxy)  ,
    cxz (value_in.cxz)  ,
    cxyz (value_in.cxyz),
    ax (value_in.ax)    ,
    ay (value_in.ay)    ,
    az (value_in.az)    ,
    axy (value_in.axy)  ,
    axz (value_in.axz)  ,
    axyz (value_in.axyz)
  {}

  Coeff &Coeff::operator= (const Coeff &r)
  {
    this -> c0   = r.c0  ;
    this -> cx   = r.cx  ;
    this -> cy   = r.cy  ;
    this -> cz   = r.cz  ;
    this -> cxy  = r.cxy ;
    this -> cxz  = r.cxz ;
    this -> cxyz = r.cxyz;
    this -> ax   = r.ax  ;
    this -> ay   = r.ay  ;
    this -> az   = r.az  ;
    this -> axy  = r.axy ;
    this -> axz  = r.axz ;
    this -> axyz = r.axyz;
    return (*this);
  }

  template<int dim>
  MMS<dim>::MMS()
    :
    Function<dim, double> (EquationComponents<dim>::n_components),
    initialized (false),
    is_NS (false)
  {}

  template<int dim>
  MMS<dim>::MMS (const MMS &mms_in)
    :
    Function<dim, double> (EquationComponents<dim>::n_components),
    initialized (mms_in.initialized),
    is_NS (mms_in.is_NS)
    // TODO: It seems now the deal.II provided std_cxx11::array does not
    // support copying initialization, so do it by 'for loop'.
  {
    for (unsigned int i=0; i<EquationComponents<dim>::n_components; ++i)
      {
        c[i] = mms_in.c[i];
      }
  }

  template<int dim>
  void MMS<dim>::reinit
  (std_cxx11::array<Coeff, EquationComponents<dim>::n_components> &c_in)
  {
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        c[ic] = c_in[ic];
      }
    initialized = true;
    return;
  }

  template<int dim>
  void MMS<dim>::set_eqn_to_NS()
  {
    is_NS = true;
    return;
  }

  template<int dim>
  void MMS<dim>::set_eqn_to_Euler()
  {
    is_NS = false;
    return;
  }

  template<int dim>
  bool MMS<dim>::is_subsonic() const
  {
    // This implementation id no reliable, but works under current situation.
    // Mean velocity square
    double velocity_magnitude_square (0.0);
    for (unsigned int d=0; d<dim; ++d)
      {
        velocity_magnitude_square += c[d].c0*c[d].c0;
      }
    return (velocity_magnitude_square < 1.0);
  }

  template<int dim>
  double MMS<dim>::value (const Point<dim>  &p,
                          const unsigned int component) const
  {
    double rv (0.0);
    std_cxx11::array<double, dim> p_array;
    for (unsigned int d=0; d<dim; ++d)
      {
        p_array[d] = p[d];
      }
    value_at_point<double> (p_array, component, rv);
    return (rv);
  }

  template<int dim>
  void MMS<dim>::vector_value (const Point<dim> &p,
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

  template<int dim>
  void MMS<dim>::value_list (const std::vector<Point<dim> > &point_list,
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

  template<int dim>
  void MMS<dim>::vector_value_list (const std::vector<Point<dim> > &point_list,
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

  template<int dim>
  void MMS<dim>::evaluate (const Point<dim>   &p,
                           F_V &value,
                           F_T &grad,
                           F_V &source,
                           const bool need_source) const
  {
    Assert (initialized, ExcMessage ("run MMS::reinit(...) before MMS::evaluation(...)."));

    std_cxx11::array<FADD, dim> p_array;
    for (unsigned int d=0; d<dim; ++d)
      {
        p_array[d] = p[d];
        p_array[d].diff (d, dim);
      }
    FADD_V value_ad;

    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        value_ad[ic] = 0.0;
      }

    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        value_at_point<FADD> (p_array,ic,value_ad[ic]);
      }

    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        value[ic] = value_ad[ic].val();
        for (unsigned int d=0; d<dim; ++d)
          {
            grad[ic][d] = value_ad[ic].dx (d);
          }
      }

    if (need_source)
      {
        {
          FADD_T flux;

          EulerEquations<dim>::compute_inviscid_flux (value_ad, flux);
          for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
            {
              source[ic] = 0;
              for (unsigned int d=0; d<dim; ++d)
                {
                  source[ic] += flux[ic][d].dx (d);
                }
            }
        }
        if (is_NS)
          {
            FADD_T grad_value;
            for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
              for (unsigned int d=0; d<dim; ++d)
                {
                  grad_value[ic][d] = value_ad[ic].dx (d);
                }

            FADD_T visc_flux;

            EulerEquations<dim>::compute_viscous_flux (value_ad, grad_value, visc_flux, 0, 0);

            for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
              for (unsigned int d=0; d<dim; ++d)
                {
                  source[ic] -= visc_flux[ic][d].dx (d);
                }
          }
      }
    return;
  }

  template<>
  template<typename Number>
  void MMS<2>::value_at_point (const std_cxx11::array<Number,2> &p,
                               const unsigned int component,
                               Number &result) const
  {
    const unsigned int dim = 2;
    const Number &x=p[0];
    const Number &y=p[1];

    const double &Pi = numbers::PI;
    const Coeff &t = c[component];
    switch (component)
      {
      case EquationComponents<dim>::first_momentum_component:
        result += t.c0;
        result += t.cx  * std::sin (t.ax  * Pi * x);
        result += t.cy  * std::cos (t.ay  * Pi * y);
        result += t.cxy * std::cos (t.axy * Pi * x * y);
        result += 0.4 * std::exp (-Pi*0.5*0.4*0.4*
                                  ((x-0.5)* (x-0.5) + (y-0.5)* (y-0.5)))
                  - 0.3111070717;
        break;
      case EquationComponents<dim>::first_momentum_component + 1:
        result += t.c0;
        result += t.cx  * std::cos (t.ax  * Pi * x);
        result += t.cy  * std::sin (t.ay  * Pi * y);
        result += t.cxy * std::cos (t.axy * Pi * x * y);
        result += 0.4 * std::exp (-Pi*0.5*0.4*0.4*
                                  ((x-0.5)* (x-0.5) + (y-0.5)* (y-0.5)))
                  - 0.3111070717;
        break;
      case EquationComponents<dim>::density_component:
        result += t.c0;
        result += t.cx  * std::sin (t.ax  * Pi * x);
        result += t.cy  * std::cos (t.ay  * Pi * y);
        result += t.cxy * std::cos (t.axy * Pi * x * y);
        result += (0.4 * std::exp (-Pi*0.5*0.4*0.4*
                                   ((x-0.5)* (x-0.5) + (y-0.5)* (y-0.5)))
                   - 0.3111070717) * 2.0;
        break;
      case EquationComponents<dim>::pressure_component:
        result += t.c0;
        result += t.cx  * std::cos (t.ax  * Pi * x);
        result += t.cy  * std::sin (t.ay  * Pi * y);
        result += t.cxy * std::sin (t.axy * Pi * x * y);
        result += (0.4 * std::exp (-Pi*0.5*0.4*0.4*
                                   ((x-0.5)* (x-0.5) + (y-0.5)* (y-0.5)))
                   - 0.3111070717) * 3.0;
        break;
      default:
        Assert (false, ExcNotImplemented());
        break;
      }
    return;
  }

  template<>
  template<typename Number>
  void MMS<3>::value_at_point (const std_cxx11::array<Number,3> &p,
                               const unsigned int component,
                               Number &result) const
  {
    const unsigned int dim = 3;
    const Number &x=p[0];
    const Number &y=p[1];
    const Number &z=p[2];

    const double &Pi = numbers::PI;
    const Coeff &t = c[component];
    switch (component)
      {
      case EquationComponents<dim>::first_momentum_component:
        result += t.c0;
        result += t.cx  * std::sin (t.ax  * Pi * x);
        result += t.cy  * std::cos (t.ay  * Pi * y);
        result += t.cz  * std::cos (t.az  * Pi * z);
        break;
      case EquationComponents<dim>::first_momentum_component + 1:
        result += t.c0;
        result += t.cx  * std::cos (t.ax  * Pi * x);
        result += t.cy  * std::sin (t.ay  * Pi * y);
        result += t.cz  * std::sin (t.az  * Pi * z);
        break;
      case EquationComponents<dim>::first_momentum_component + 2:
        result += t.c0;
        result += t.cx  * std::sin (t.ax  * Pi * x);
        result += t.cy  * std::sin (t.ay  * Pi * y);
        result += t.cz  * std::cos (t.az  * Pi * z);
        break;
      case EquationComponents<dim>::density_component:
        result += t.c0;
        result += t.cx  * std::sin (t.ax  * Pi * x);
        result += t.cy  * std::cos (t.ay  * Pi * y);
        result += t.cz  * std::sin (t.az  * Pi * z);
        break;
      case EquationComponents<dim>::pressure_component:
        result += t.c0;
        result += t.cx  * std::cos (t.ax  * Pi * x);
        result += t.cy  * std::sin (t.ay  * Pi * y);
        result += t.cz  * std::cos (t.az  * Pi * z);
        break;
      default:
        Assert (false, ExcNotImplemented());
        break;
      }

    return;
  }


  // Explicit instantiations
  template class MMS<2>;
  template class MMS<3>;
}
