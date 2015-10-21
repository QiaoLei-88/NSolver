//
//  Created by 乔磊 on 15/10/21.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#ifndef __NSolver__StabilizationParameters_h__
#define __NSolver__StabilizationParameters_h__

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <NSolver/EquationComponents.h>

namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    template <int dim>
    struct StabilizationParameters
    {
      enum DiffusionType
      {
        diffu_entropy,
        diffu_cell_size,
        diffu_const
      };

      DiffusionType diffusion_type;
      double diffusion_power;
      double diffusion_coefficoent;
      double entropy_visc_cE;
      double entropy_visc_cLinear;
      double diffusion_factor[EquationComponents<dim>::n_components];
      double laplacian_continuation;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };
  }
}
#endif
