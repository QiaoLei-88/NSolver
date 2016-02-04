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
        prm.declare_entry ("entropy use global h min", "false",
                           Patterns::Bool(),
                           "Use global minimum cell size for entropy viscosity");
        prm.declare_entry ("entropy with rho max", "true",
                           Patterns::Bool(),
                           "Scale entropy viscosity with local maximum density");
        prm.declare_entry ("entropy with solution diff", "true",
                           Patterns::Bool(),
                           "Compute entropy viscosity with solution difference");

        prm.declare_entry ("oscillation visc ceiling", "-3",
                           Patterns::Double(),
                           "Ceiling value of oscillation indicator to evaluate artificial viscosity");
        prm.declare_entry ("oscillation visc ground", "-4",
                           Patterns::Double(),
                           "Ceiling value of oscillation indicator to evaluate artificial viscosity");
        prm.declare_entry ("oscillation visc background", "1.0e-8",
                           Patterns::Double(),
                           "Background value of artificial viscosity for under ground oscillation");
        prm.declare_entry ("oscillation visc coefficient", "1.0",
                           Patterns::Double(),
                           "Scale factor of artificial viscosity for under ground oscillation");

        prm.declare_entry ("smooth artificial viscosity", "false",
                           Patterns::Bool(),
                           "Smooth artificial viscosity via setting value to maximum among its neighbors.");

        prm.declare_entry ("continuation type", "timeCFL",
                           Patterns::Anything(),
                           "continuation type");

        prm.declare_entry ("use local laplacian coefficient", "false",
                           Patterns::Bool(),
                           "use local (cell size related) laplacian coefficient");

        prm.declare_entry ("CountSolDiffInRes", "true",
                           Patterns::Bool(),
                           "Count solution difference term in residual");
        prm.declare_entry ("CountArtiViscInResPhy", "false",
                           Patterns::Bool(),
                           "Count artificial viscosity term in physical residual");
        prm.declare_entry ("use conservative variables for time difference", "true",
                           Patterns::Bool(),
                           "Use conservative variables for time difference in time marching");

        prm.declare_entry ("continuation min decrease rate","0.5",
                           Patterns::Double(),
                           "Ratio that continuation coefficient at least to decrease on each step");
        prm.declare_entry ("continuation decrease residual power","2.0",
                           Patterns::Double(),
                           "Power on relative residual for computing new continuation coefficient");
        prm.declare_entry ("continuation blend starting ratio","40.0",
                           Patterns::Double(),
                           "Staring Laplacian to time ratio of blending continuation");
        prm.declare_entry ("switch threshold","10.0",
                           Patterns::Double(),
                           "Ratio threshold to switch from laplacian to time continuation");

        prm.declare_entry ("Laplacian continuation", "-1.0",
                           Patterns::Double(),
                           "Coefficient for Laplacian continuation");
        prm.declare_entry ("compute laplacian coeff from Mach max", "false",
                           Patterns::Bool(),
                           "Compute laplacian coefficient from maximum Mach number among entire flow field");
        prm.declare_entry ("enable partial laplacian", "false",
                           Patterns::Bool(),
                           "Enable partial laplacian");
        prm.declare_entry ("dodge mesh adaptation", "false",
                           Patterns::Bool(),
                           "Avoid doing mesh refinement and decreasing Laplacian continuation on same time step.");
        prm.declare_entry ("Laplacian zero", "1e-9",
                           Patterns::Double(),
                           "Threshold of Laplacian Coefficient to be force to zero");
        prm.declare_entry ("Laplacian Newton tolerance", "0.1",
                           Patterns::Double (0.0),
                           "Threshold of Laplacian Coefficient to be force to zero");

        prm.declare_entry ("SUPG factor", "-1.0",
                           Patterns::Double(),
                           "SUPG stabilization factor.");

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
        {
          const std::string prm_buf = prm.get ("diffusion type");
          if (prm_buf == "entropy")
            {
              diffusion_type = diffu_entropy;
            }
          else if (prm_buf == "entropy_DRB")
            {
              diffusion_type = diffu_entropy_DRB;
            }
          else if (prm_buf == "oscillation")
            {
              diffusion_type = diffu_oscillation;
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
        }
        diffusion_power = prm.get_double ("diffusion power");
        diffusion_coefficoent = prm.get_double ("diffusion coefficient");
        entropy_visc_cE = prm.get_double ("entropy visc cE");
        entropy_visc_cLinear = prm.get_double ("entropy visc cLinear");
        entropy_use_global_h_min = prm.get_bool ("entropy use global h min");
        entropy_with_rho_max = prm.get_bool ("entropy with rho max");
        entropy_with_solution_diff = prm.get_bool ("entropy with solution diff");

        oscillation_visc_ceiling = prm.get_double ("oscillation visc ceiling");
        oscillation_visc_ground = prm.get_double ("oscillation visc ground");
        oscillation_visc_background = prm.get_double ("oscillation visc background");
        oscillation_visc_coefficient = prm.get_double ("oscillation visc coefficient");

        smooth_artificial_viscosity = prm.get_bool ("smooth artificial viscosity");

        {
          const std::string prm_buf = prm.get ("continuation type");
          if (prm_buf == "timeCFL")
            {
              continuation_type = CT_timeCFL;
            }
          else if (prm_buf == "timeCFL2")
            {
              continuation_type = CT_timeCFL2;
            }
          else if (prm_buf == "time")
            {
              continuation_type = CT_time;
            }
          else if (prm_buf == "laplacian")
            {
              continuation_type = CT_laplacian;
            }
          else if (prm_buf == "switch")
            {
              continuation_type = CT_switch;
            }
          else if (prm_buf == "alternative")
            {
              continuation_type = CT_alternative;
            }
          else if (prm_buf == "blend")
            {
              continuation_type = CT_blend;
            }
          else
            {
              AssertThrow (false, ExcNotImplemented());
            }
        }
        continuation_min_decrease_rate = prm.get_double ("continuation min decrease rate");
        continuation_decrease_residual_power = prm.get_double ("continuation decrease residual power");
        continuation_blend_starting_ratio = prm.get_double ("continuation blend starting ratio");
        continuation_switch_threshold = prm.get_double ("switch threshold");

        use_local_laplacian_coefficient
          = prm.get_bool ("use local laplacian coefficient");

        count_solution_diff_in_residual
          = prm.get_bool ("CountSolDiffInRes");

        count_artificial_visc_term_in_phisical_residual
          = prm.get_bool ("CountArtiViscInResPhy");

        use_conservative_variables_for_time_diff
          = prm.get_bool ("use conservative variables for time difference");

        laplacian_continuation = prm.get_double ("Laplacian continuation");
        compute_laplacian_coeff_from_Mach_max = prm.get_bool ("compute laplacian coeff from Mach max");
        enable_partial_laplacian = prm.get_bool ("enable partial laplacian");
        dodge_mesh_adaptation = prm.get_bool ("dodge mesh adaptation");
        laplacian_zero = prm.get_double ("Laplacian zero");
        laplacian_newton_tolerance = prm.get_double ("Laplacian Newton tolerance");

        for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
          {
            diffusion_factor[di] = prm.get_double ("diffusion factor for w_"
                                                   + Utilities::int_to_string (di));
          }

        SUPG_factor = prm.get_double ("SUPG factor");
      }
      prm.leave_subsection();
      return;
    }

    template struct StabilizationParameters<2>;
    template struct StabilizationParameters<3>;
  }
}
