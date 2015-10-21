//
//  Created by 乔磊 on 15/10/21.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#include <NSolver/Parameters/StabilizationParameters.h>

namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    template <int dim>
    void StabilizationParameters<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("stabilization parameters");
      {
        prm.declare_entry ("diffusion type", "cell size",
                           Patterns::Anything(),
                           "How to calculate diffusion");

        prm.declare_entry ("diffusion power", "2.0",
                           Patterns::Double(),
                           "power of mesh size for diffusion");

        prm.declare_entry ("diffusion coefficient", "1.0",
                           Patterns::Double(),
                           "predefined diffusion coefficient");
        prm.declare_entry ("entropy visc cE", "1.0",
                           Patterns::Double (0.0),
                           "Scale factor on entropy viscosity");
        prm.declare_entry ("entropy visc cLinear", "0.25",
                           Patterns::Double (0.0),
                           "Scale factor on linear limit of entropy viscosity");
        prm.declare_entry ("Laplacian continuation", "-1.0",
                           Patterns::Double(),
                           "Coefficient for Laplacian continuation");
        prm.declare_entry ("Laplacian zero", "1e-9",
                           Patterns::Double(),
                           "Threshold of Laplacian Coefficient to be force to zero");
        prm.declare_entry ("Laplacian Newton tolerance", "0.1",
                           Patterns::Double (0.0),
                           "Threshold of Laplacian Coefficient to be force to zero");

        for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
          {
            prm.declare_entry ("diffusion factor for w_" + Utilities::int_to_string (di),
                               "1.0",
                               Patterns::Double (0),
                               "diffusion factor for component w_" + Utilities::int_to_string (di));
          }
      }
      prm.leave_subsection();
      return;
    }

    template <int dim>
    void StabilizationParameters<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("stabilization parameters");
      {
        const std::string prm_buf = prm.get ("diffusion type");
        if (prm_buf == "entropy")
          {
            diffusion_type = diffu_entropy;
          }
        else if (prm_buf == "cell size")
          {
            diffusion_type = diffu_cell_size;
          }
        else if (prm_buf == "const")
          {
            diffusion_type = diffu_const;
          }
        else
          {
            AssertThrow (false, ExcNotImplemented());
          }

        diffusion_power = prm.get_double ("diffusion power");
        diffusion_coefficoent = prm.get_double ("diffusion coefficient");
        entropy_visc_cE = prm.get_double ("entropy visc cE");
        entropy_visc_cLinear = prm.get_double ("entropy visc cLinear");
        laplacian_continuation = prm.get_double ("Laplacian continuation");
        laplacian_zero = prm.get_double ("Laplacian zero");
        laplacian_newton_tolerance = prm.get_double ("Laplacian Newton tolerance");

        for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
          {
            diffusion_factor[di] = prm.get_double ("diffusion factor for w_"
                                                   + Utilities::int_to_string (di));
          }
      }
      prm.leave_subsection();
      return;
    }

    template struct StabilizationParameters<2>;
    template struct StabilizationParameters<3>;
  }
}
