//
//  Created by 乔磊 on 15/4/30.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__FEParameters__
#define __NSolver__FEParameters__

#include <deal.II/base/parameter_handler.h>

namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    struct FEParameters
    {
      FEParameters();
      FEParameters(const FEParameters &para_in);
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);

      enum MappingType
      {
        MappingQ,
        MappingC1
      };
      MappingType mapping_type;

      int fe_degree;
      int mapping_degree;
      int quadrature_degree;
      int face_quadrature_degree;
      int error_quadrature_degree;
    };

  } // namespace Parameters
} // namespace NSFEMSolver

#endif
