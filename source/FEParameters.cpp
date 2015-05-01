//
//  Created by 乔磊 on 15/4/30.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include "FEParameters.h"

namespace NSolver
{
  namespace Parameters
  {
    using namespace dealii;
    FEParameters::FEParameters()
    {}

    FEParameters::FEParameters (const FEParameters &para_in)
    {
      fe_degree               = para_in.fe_degree              ;
      mapping_degree          = para_in.mapping_degree         ;
      quadrature_degree       = para_in.quadrature_degree      ;
      face_quadrature_degree  = para_in.face_quadrature_degree ;
      error_quadrature_degree = para_in.error_quadrature_degree;
    }

    void FEParameters::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("FE Parameters");
      {

        prm.declare_entry ("fe degree", "1",
                           Patterns::Integer(),
                           "Element degree");

        prm.declare_entry ("mapping degree", "-1",
                           Patterns::Integer(),
                           "Mapping degree");

        prm.declare_entry ("quadrature degree", "-1",
                           Patterns::Integer(),
                           "Cell quadrature degree");

        prm.declare_entry ("face quadrature_degree", "-1",
                           Patterns::Integer(),
                           "Face quadrature degree");

        prm.declare_entry ("error quadrature_degree", "-1",
                           Patterns::Integer(),
                           "Quadrature degree for error evaluation");
      }
      prm.leave_subsection();
    }


    void FEParameters::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("FE Parameters");
      {
        fe_degree = prm.get_integer ("fe degree");
        if (fe_degree <= 0)
          {
            fe_degree = 1;
          }
        mapping_degree = prm.get_integer ("mapping degree");
        if (mapping_degree <= 0)
          {
            mapping_degree = fe_degree;
          }

        quadrature_degree = prm.get_integer ("quadrature degree");
        if (quadrature_degree <= 0)
          {
            quadrature_degree = fe_degree + 1;
          }

        face_quadrature_degree = prm.get_integer ("face quadrature_degree");
        if (face_quadrature_degree <= 0)
          {
            face_quadrature_degree = quadrature_degree;
          }

        error_quadrature_degree = prm.get_integer ("error quadrature_degree");
        if (error_quadrature_degree <= 0)
          {
            error_quadrature_degree = quadrature_degree + 1;
          }
      }
      prm.leave_subsection();
    }

  }
}
