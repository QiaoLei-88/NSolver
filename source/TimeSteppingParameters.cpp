//
//  TimeSteppingParameters.cpp
//  NSolver
//
//  Created by 乔磊 on 15/5/18.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include "TimeSteppingParameters.h"

namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    void TimeStepping::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("time stepping");
      {
        prm.declare_entry ("final time", "10.0",
                           Patterns::Double (0),
                           "simulation end time");
        prm.declare_entry ("max Newton iter", "10000",
                           Patterns::Integer (0),
                           "maximum number of Newton interations");
        prm.declare_entry ("time march tolerance", "-10.0",
                           Patterns::Double(),
                           "Terminate time marching when log10 of error L2 norm is less than this value");


        prm.declare_entry ("theta scheme value", "0.5",
                           Patterns::Double (0,1),
                           "value for theta that interpolated between explicit "
                           "Euler (theta=0), Crank-Nicolson (theta=0.5), and "
                           "implicit Euler (theta=1).");


        prm.declare_entry ("rigid reference time step", "true",
                           Patterns::Bool(),
                           "whether use specified reference time step or"
                           "calulate according to CFL condition.");
        prm.declare_entry ("reference time step", "0.1",
                           Patterns::Double(),
                           "simulation time step");


        prm.declare_entry ("CFL number", "1.0",
                           Patterns::Double (0),
                           "CFL number");
        prm.declare_entry ("unsteady CFL number max", "-1.0",
                           Patterns::Double(),
                           "maximum of CFL number for unsteady simulation");
        prm.declare_entry ("CFL number min", "-1.0",
                           Patterns::Double(),
                           "minimum of CFL number");
        prm.declare_entry ("allow increase CFL", "true",
                           Patterns::Bool(),
                           "allow increase CFL number for unsteady simulation");
        prm.declare_entry ("allow decrease CFL", "true",
                           Patterns::Bool(),
                           "allow decrease CFL number");


        prm.declare_entry ("iter in stage1", "5",
                           Patterns::Integer (0),
                           "number of interations in stage one for time step increasing in steady case");
        prm.declare_entry ("step increasing ratio stage1", "1.0",
                           Patterns::Double (0),
                           "step increasing ratio in stage1 one for time step increasing in steady case");
        prm.declare_entry ("minimum step increasing ratio stage2", "0.1",
                           Patterns::Double (0),
                           "minimum step increasing ratio in stage two for time step increasing in steady case");
        prm.declare_entry ("step increasing power stage2", "1.2",
                           Patterns::Double (0),
                           "step increasing power in stage two for time step increasing in steady case");


        prm.declare_entry ("solution extrapolation length", "1.0",
                           Patterns::Double(),
                           "relative length of the forward extrapolation for predicting solution of next time step");


        prm.declare_entry ("newton linear search length try limit", "0",
                           Patterns::Integer (0,8),
                           "largest index of linear search length could try. "
                           "value of linear search length is defined internally.");
      }
      prm.leave_subsection();
    }


    void TimeStepping::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("time stepping");
      {
        final_time = prm.get_double ("final time");
        max_Newton_iter = prm.get_integer ("max Newton iter");
        time_march_tolerance = prm.get_double ("time march tolerance");

        theta = prm.get_double ("theta scheme value");

        rigid_reference_time_step = prm.get_bool ("rigid reference time step");
        reference_time_step = prm.get_double ("reference time step");
        AssertThrow (reference_time_step!=0.0, ExcMessage (" Time step size cann't be 0."));
        is_steady = false;
        if (reference_time_step <= 0)
          {
            is_steady = true;
            reference_time_step = -reference_time_step;
          }


        CFL_number = prm.get_double ("CFL number");
        CFL_number_max = prm.get_double ("unsteady CFL number max");
        if (CFL_number_max < CFL_number)
          {
            CFL_number_max = CFL_number;
          }
        CFL_number_min = prm.get_double ("CFL number min");
        if (CFL_number_min < 0.0 || CFL_number_min > CFL_number)
          {
            // By default, allow half CFL number ten times.
            // 1/1024 < 0.0005 < 1/2048
            CFL_number_min = 0.0005 * CFL_number;
          }
        allow_increase_CFL = prm.get_bool ("allow increase CFL");
        allow_decrease_CFL = prm.get_bool ("allow decrease CFL");

        n_iter_stage1 = prm.get_integer ("iter in stage1");
        step_increasing_ratio_stage1 = prm.get_double ("step increasing ratio stage1");
        minimum_step_increasing_ratio_stage2 = prm.get_double ("minimum step increasing ratio stage2");
        step_increasing_power_stage2 = prm.get_double ("step increasing power stage2");


        solution_extrapolation_length = prm.get_double ("solution extrapolation length");

        newton_linear_search_length_try_limit = prm.get_integer ("newton linear search length try limit");
      }
      prm.leave_subsection();
    }

  } // End namespace Parameters
} // End namespace NSFEMSolver
