//
//  MMS.cpp
//  NSolver
//
//  Created by 乔磊 on 15/4/24.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include "NSEquation.h"

namespace NSolver
{

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
    :initialized (false),
     is_NS (false)
  {}


  void MMS::reinit
  (std_cxx11::array<Coeff_2D, EulerEquations<dim2>::n_components> &c_in)
  {
    for (unsigned int ic=0; ic<EulerEquations<dim2>::n_components; ++ic)
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

  void MMS::evaluate (const Point<dim2>   &p,
                      std_cxx11::array<double, EulerEquations<dim2>::n_components> &value,
                      std_cxx11::array<double, EulerEquations<dim2>::n_components> &source,
                      const bool need_source) const
  {
    Assert (initialized, ExcMessage ("run MMS::reinit(...) before MMS::evaluation(...)."));

    typedef Sacado::Fad::DFad<double> FADD;
    FADD x=p (0);
    FADD y=p (1);
    x.diff (0,2);
    y.diff (1,2);
    const double &Pi = numbers::PI;
    std_cxx11::array<FADD, EulerEquations<dim2>::n_components> value_ad;
    for (unsigned int ic=0; ic<EulerEquations<dim2>::n_components; ++ic)
      {
        value_ad[ic] = 0.0;
      }

    for (unsigned int ic=0; ic<EulerEquations<dim2>::n_components; ++ic)
      {
        const Coeff_2D &t = c[ic];
        switch (ic)
          {
          case EulerEquations<dim2>::first_momentum_component:
            value_ad[ic] += t.c0;
            value_ad[ic] += t.cx  * std::sin (t.ax  * Pi * x);
            value_ad[ic] += t.cy  * std::cos (t.ay  * Pi * y);
            value_ad[ic] += t.cxy * std::cos (t.axy * Pi * x * y);
            break;
          case EulerEquations<dim2>::first_momentum_component + 1:
            value_ad[ic] += t.c0;
            value_ad[ic] += t.cx  * std::cos (t.ax  * Pi * x);
            value_ad[ic] += t.cy  * std::sin (t.ay  * Pi * y);
            value_ad[ic] += t.cxy * std::cos (t.axy * Pi * x * y);
            break;
          case EulerEquations<dim2>::density_component:
            value_ad[ic] += t.c0;
            value_ad[ic] += t.cx  * std::sin (t.ax  * Pi * x);
            value_ad[ic] += t.cy  * std::cos (t.ay  * Pi * y);
            value_ad[ic] += t.cxy * std::cos (t.axy * Pi * x * y);
            break;
          case EulerEquations<dim2>::pressure_component:
            value_ad[ic] += t.c0;
            value_ad[ic] += t.cx  * std::cos (t.ax  * Pi * x);
            value_ad[ic] += t.cy  * std::sin (t.ay  * Pi * y);
            value_ad[ic] += t.cxy * std::sin (t.axy * Pi * x * y);
            break;
          default:
            Assert (false, ExcNotImplemented());
            break;
          }
      }

    for (unsigned int ic=0; ic<EulerEquations<dim2>::n_components; ++ic)
      {
        value[ic] = value_ad[ic].val();
      }

    if (need_source)
      {
        {
          std_cxx11::array
          <std_cxx11::array <FADD, dim2>, EulerEquations<dim2>::n_components >
          flux;

          EulerEquations<dim2>::compute_inviscid_flux (value_ad, flux);

          for (unsigned int ic=0; ic<EulerEquations<dim2>::n_components; ++ic)
            {
              source[ic] = flux[ic][0].dx (0) + flux[ic][1].dx (1);
            }
        }
        if (is_NS)
          {
            std_cxx11::array
            <std_cxx11::array <FADD, dim2>, EulerEquations<dim2>::n_components >
            grad_value;
            for (unsigned int ic=0; ic<EulerEquations<dim2>::n_components; ++ic)
              {
                grad_value[ic][0] = value_ad[ic].dx (0);
                grad_value[ic][1] = value_ad[ic].dx (1);
              }

            std_cxx11::array
            <std_cxx11::array <FADD, dim2>, EulerEquations<dim2>::n_components >
            visc_flux;

            EulerEquations<dim2>::compute_viscous_flux (value_ad, grad_value, visc_flux);

            for (unsigned int ic=0; ic<EulerEquations<dim2>::n_components; ++ic)
              {
                source[ic] -= (visc_flux[ic][0].dx (0) + visc_flux[ic][1].dx (1));
              }
          }
      }
    return;
  }
}
