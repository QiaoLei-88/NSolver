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
        /**
         * Refer to
         * Marc O. Delchini and Jean C. Ragusa and Ray A. Berry,
         * Entropy-based viscous regularization for the multi-dimensional
         * Euler equations in low-Mach and transonic flows,
         * Computers & Fluids 118 (2015) 225–244.
         * for details.
         */
        diffu_entropy_DRB,
        diffu_cell_size,
        diffu_const
      };
      DiffusionType diffusion_type;

      enum ContinuationType
      {
        CT_timeCFL,
        CT_time,
        CT_laplacian,
        CT_switch,
        CT_alternative,
        CT_blend
      };
      ContinuationType continuation_type;
      double continuation_switch_threshold;

      double diffusion_power;
      double diffusion_coefficoent;
      double entropy_visc_cE;
      double entropy_visc_cLinear;
      bool   entropy_use_global_h_min;
      double diffusion_factor[EquationComponents<dim>::n_components];
      double laplacian_continuation;
      /**
       * Use laplacian continuation only in region with large indicators.
       */
      bool enable_partial_laplacian;
      /**
       * When this flag is set, laplacian refinement and mesh adaptation will be
       * done alternatively to increase stability when they are both enabled.
       * Default value is false.
       */
      bool dodge_mesh_adaptation;

      /**
       * When doing Laplacian continuation, the Laplacian coefficient will be
       * set to zero if its value is less than this parameter.
       */
      double laplacian_zero;

      /**
       * If Newton iteration achieved a convergence rate larger or equal than
       * this parameter, it be recognized as quadratic convergence.
       */
      double laplacian_newton_quadratic;

      /**
       * Relative Newton iteration tolerance comparing with Laplacian coefficient.
       */
      double laplacian_newton_tolerance;

      /**
       * If Newton iteration achieved quadratic convergence, decrease
       * Laplacian coefficient with this rate.
       */
      double laplacian_decrease_rate;

      /**
       * SUPG stabilization factor. A non-positive value will disable the SUPG term.
       */
      double SUPG_factor;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };
  }
}
#endif
