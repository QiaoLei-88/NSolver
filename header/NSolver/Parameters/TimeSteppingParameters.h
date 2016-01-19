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
       * If true, not time term will be added to linear system.
       * This is used when other continuation method such as Laplacian continuation is employed.
       * By default the value is false.
       */
      bool turn_off_time_marching;

      /**
       * Parameters for when to terminate time stepping.
       * Unsteady run only terminates when final time is arrived.
       * Steady run terminates when number of Newton iterations reaches
       * <code>max_Newton_iter</code> or time marching L2 norm is
       * below <code>solution_update_l2_tolerance<code>.
       */
      double final_time;
      unsigned int max_Newton_iter;
      double solution_update_l2_tolerance;

      /**
       * The tolerance to end Newton iteration. This parameter will be interpreted
       * into different means according to its sign. If a positive value is passed in,
       * it will be used as absolute tolerance. If a negative value is passed in,
       * it will be used as a relative order to drop.
       */
      double nonlinear_tolerance;

      /**
       * Parameter for residual blending of current and last time step.
       * <code>theta</code> = 1.0 stands for implicit Euler scheme;
       * <code>theta</code> = 0.5 stands for Crank-Nicolson scheme;
       * <code>theta</code> = 1.0 stands for explicit Euler scheme.
       */
      double theta;

      /**
       * If <code>rigid_reference_time_step</code> is true, reference time step size
       * is set to the value specified in the input file, otherwise the reference
       * time step size is calculated according to CFL condition.
       * This flag is set to true by default.
       */
      bool rigid_reference_time_step;
      /**
       * Negative <code>reference_time_step</code> will be recognized as steady
       * simulation. For steady cases, CFL number is evaluated in a special way.
       *
       * If reference_time_step is provided as zero, an exception will be thrown.
       */
      double reference_time_step;
      bool is_steady;

      /**
       * Flag to use local time step size. Only effective in steady case.
       */
      bool use_local_time_step_size;

      /**
       * parameters for evaluating CFL number
       */
      double CFL_number;
      /**
       * Note: For steady simulation, maximum of CFL number is only limited by
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
