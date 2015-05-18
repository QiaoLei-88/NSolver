//
//  TimeSteppingParameters.h
//  NSolver
//
//  Created by 乔磊 on 15/5/18.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__TimeSteppingParameters__
#define __NSolver__TimeSteppingParameters__

#include <deal.II/base/parameter_handler.h>

namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    struct TimeStepping
    {
      /**
       * Parameters for when to terminate time stepping
       */
      double final_time;
      double time_march_tolerance;

      /**
       * Parameter for residual blending of current and last time step.
       * <code>theta</code> = 1.0 stands for implicit Euler scheme;
       * <code>theta</code> = 0.5 stands for Crank-Nicolson scheme;
       * <code>theta</code> = 1.0 stands for explicit Euler scheme.
       */
      double theta;

      /**
       * If <code>rigid_reference_time_step</code> is ture, reference time step size
       * is set to the value specified in the input file, otherwise the reference
       * time step size is calculated according to CFL condition.
       * This flag is set to true by default.
       */
      bool rigid_reference_time_step;
      /**
       * Negative <code>reference_time_step</code> will be recognized as steady
       * simulation. For stationary cases, CFL number is evaluated in a special way.
       *
       * If reference_time_step is provided as zero, an exception will be thrown.
       */
      double reference_time_step;
      bool is_stationary;

      /**
       * parameters for evaluating CFL number
       */
      double CFL_number;
      /**
       * Note: For steady simulation, maximun of CFL number is only limited by
       * range of <code>double</code> type.
       */
      double CFL_number_max;
      double CFL_number_min;
      /**
       * Whether allow increase CFL number in unsteady simulation.
       */
      bool allow_increase_CFL;
      /**
       * Whether allow decrease CFL number when newton iteration diverged.
       */
      bool allow_decrease_CFL;

      /**
       * Parameters for evaluate CFL number in steady simulation.
       * Please refer to John Gatsis's Phd thesis for details.
       */
      unsigned int n_iter_stage1;
      double step_increasing_ratio_stage1;
      double minimum_step_increasing_ratio_stage2;
      double step_increasing_power_stage2;


      /**
       * Predict solution of next time step by making a linear extrapolation from current
       * and last time step. This parameter controls the relative length of the
       * forward extrapolation. Specifically,
       * predicted_solution =  current_solution * (1+solution_extrapolation_length)
       *                      -old_solution * solution_extrapolation_length;
       */
      bool solution_extrapolation_length;

      /**
       * There is an array defined in NSolver::run() that contains a list of
       * linear search length. This parameter is the maximum index of this array
       * allow to try. Default value of this parameter is 0.
       */
      int newton_linear_search_length_try_limit;


      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);

    };
  }
}

#endif /* defined(__NSolver__TimeSteppingParameters__) */
