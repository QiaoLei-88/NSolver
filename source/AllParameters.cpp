//
//  AllParameters.cpp
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include "AllParameters.h"

namespace NSFEMSolver
{
  namespace Parameters
  {

    using namespace dealii;

    void Solver::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("linear solver");
      {
        prm.declare_entry ("output", "quiet",
                           Patterns::Selection ("quiet|verbose"),
                           "State whether output from solver runs should be printed. "
                           "Choices are <quiet|verbose>.");
        prm.declare_entry ("method", "gmres",
                           Patterns::Selection ("gmres|direct"),
                           "The kind of solver for the linear system. "
                           "Choices are <gmres|direct>.");
        prm.declare_entry ("residual", "1e-10",
                           Patterns::Double(),
                           "Linear solver residual");
        prm.declare_entry ("max iters", "300",
                           Patterns::Integer(),
                           "Maximum solver iterations");

        prm.declare_entry ("RCM reorder", "false",
                           Patterns::Bool(),
                           "Do Reverse Cuthill–McKee reordering.");
        prm.declare_entry ("ilut fill", "2",
                           Patterns::Double(),
                           "Ilut preconditioner fill");
        prm.declare_entry ("ilut absolute tolerance", "1e-9",
                           Patterns::Double(),
                           "Ilut preconditioner tolerance");
        prm.declare_entry ("ilut relative tolerance", "1.1",
                           Patterns::Double(),
                           "Ilut relative tolerance");
        prm.declare_entry ("ilut drop tolerance", "1e-10",
                           Patterns::Double(),
                           "Ilut drop tolerance");
      }
      prm.leave_subsection();
    }


    void Solver::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("linear solver");
      {
        const std::string op = prm.get ("output");
        if (op == "verbose")
          {
            output = verbose;
          }
        if (op == "quiet")
          {
            output = quiet;
          }

        const std::string sv = prm.get ("method");
        if (sv == "direct")
          {
            solver = direct;
          }
        else if (sv == "gmres")
          {
            solver = gmres;
          }

        linear_residual = prm.get_double ("residual");
        max_iterations  = prm.get_integer ("max iters");

        // Do not relay ont C++ standard
        AZ_RCM_reorder  = (prm.get_bool ("RCM reorder")?1:0);
        ilut_fill       = prm.get_double ("ilut fill");
        ilut_atol       = prm.get_double ("ilut absolute tolerance");
        ilut_rtol       = prm.get_double ("ilut relative tolerance");
        ilut_drop       = prm.get_double ("ilut drop tolerance");
      }
      prm.leave_subsection();
    }

//Refinement
    void Refinement::declare_parameters (ParameterHandler &prm)
    {

      prm.enter_subsection ("refinement");
      {
        prm.declare_entry ("refinement", "true",
                           Patterns::Bool(),
                           "Whether to perform mesh refinement or not");
        prm.declare_entry ("refinement fraction", "0.1",
                           Patterns::Double(),
                           "Fraction of high refinement");
        prm.declare_entry ("unrefinement fraction", "0.1",
                           Patterns::Double(),
                           "Fraction of low unrefinement");
        prm.declare_entry ("max elements", "1000000",
                           Patterns::Double(),
                           "maximum number of elements");
        prm.declare_entry ("shock value", "4.0",
                           Patterns::Double(),
                           "value for shock indicator");
        prm.declare_entry ("shock levels", "3.0",
                           Patterns::Double(),
                           "number of shock refinement levels");
      }
      prm.leave_subsection();
    }


    void Refinement::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("refinement");
      {
        do_refine     = prm.get_bool ("refinement");
        shock_val     = prm.get_double ("shock value");
        shock_levels  = prm.get_double ("shock levels");
      }
      prm.leave_subsection();
    }


    //Flux
    template<int dim>
    void Flux<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("flux");
      {
        prm.declare_entry ("flux type", "LaxFriedrichs",
                           Patterns::Selection ("LaxFriedrichs|Roe"),
                           "Numerical flux type");
        prm.declare_entry ("flux type switch to", "Roe",
                           Patterns::Selection ("LaxFriedrichs|Roe"),
                           "Numerical flux type");

        prm.declare_entry ("stab", "mesh",
                           Patterns::Selection ("constant|mesh"),
                           "Whether to use a constant stabilization parameter or "
                           "a mesh-dependent one");
        prm.declare_entry ("stab value", "1",
                           Patterns::Double(),
                           "alpha stabilization");

        prm.declare_entry ("tolerance to switch flux", "-1000",
                           Patterns::Double(),
                           "Switch flux type when log10 of time march tolerance less than this");
      }
      prm.leave_subsection();
    }

    template<int dim>
    void Flux<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("flux");
      {
        {
          const std::string buff = prm.get ("flux type");
          if (buff == "LaxFriedrichs")
            {
              numerical_flux_type = NumericalFlux::Type::LaxFriedrichs;
            }
          else if (buff == "Roe")
            {
              numerical_flux_type = NumericalFlux::Type::Roe;
            }
          else
            {
              AssertThrow (false, ExcNotImplemented());
            }
        }
        {
          const std::string buff = prm.get ("flux type switch to");
          if (buff == "LaxFriedrichs")
            {
              flux_type_switch_to = NumericalFlux::Type::LaxFriedrichs;
            }
          else if (buff == "Roe")
            {
              flux_type_switch_to = NumericalFlux::Type::Roe;
            }
          else
            {
              AssertThrow (false, ExcNotImplemented());
            }
        }
        {
          const std::string stab = prm.get ("stab");
          if (stab == "constant")
            {
              stabilization_kind = constant;
            }
          else if (stab == "mesh")
            {
              stabilization_kind = mesh_dependent;
            }
          else
            {
              AssertThrow (false, ExcNotImplemented());
            }
        }

        stabilization_value = prm.get_double ("stab value");
        tolerance_to_switch_flux = prm.get_double ("tolerance to switch flux");
      }
      prm.leave_subsection();
    }

    //Output
    void Output::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("output");
      {
        prm.declare_entry ("schlieren plot", "true",
                           Patterns::Bool(),
                           "Whether or not to produce schlieren plots");
        prm.declare_entry ("step", "-1",
                           Patterns::Double(),
                           "Output once per this period");
      }
      prm.leave_subsection();
    }



    void Output::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("output");
      {
        schlieren_plot = prm.get_bool ("schlieren plot");
        output_step = prm.get_double ("step");
      }
      prm.leave_subsection();
    }



    template <int dim>
    AllParameters<dim>::BoundaryConditions::BoundaryConditions()
      :
      values (EquationComponents<dim>::n_components)
    {}


    template <int dim>
    AllParameters<dim>::AllParameters()
      :
      initial_conditions (EquationComponents<dim>::n_components)
    {}


    template <int dim>
    void
    AllParameters<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.declare_entry ("MMS", "0",
                         Patterns::Integer(),
                         "Number of manufactured solution to test aginst");

      prm.declare_entry ("mesh", "grid.inp",
                         Patterns::Anything(),
                         "intput file name");

      prm.declare_entry ("mesh format", "gmsh",
                         Patterns::Anything(),
                         "intput mesh file format");

      prm.declare_entry ("scale mesh", "1.0",
                         Patterns::Double(),
                         "scale mesh uniformly");

      prm.declare_entry ("global refinement", "0",
                         Patterns::Integer(),
                         "level of global refinement before run");

      prm.declare_entry ("time history", "time_history.out",
                         Patterns::Anything(),
                         "output file for time history.");

      prm.declare_entry ("iteration history", "iter_history.out",
                         Patterns::Anything(),
                         "output file for all iteration history.");


      prm.declare_entry ("diffusion type", "cell size",
                         Patterns::Anything(),
                         "How to calculate diffusion");

      prm.declare_entry ("diffusion power", "2.0",
                         Patterns::Double(),
                         "power of mesh size for diffusion");

      prm.declare_entry ("diffusion coefficient", "0.001",
                         Patterns::Double(),
                         "predefined diffusion coefficient");

      prm.declare_entry ("gravity", "-1.0",
                         Patterns::Double(),
                         "gravity");

      prm.enter_subsection ("time stepping");
      {
        prm.declare_entry ("rigid reference time step", "true",
                           Patterns::Bool(),
                           "whether use specified reference time step or"
                           "calulate according to CFL condition.");

        prm.declare_entry ("auto CFL number", "true",
                           Patterns::Bool(),
                           "allow double time step size when consecutive convenvgence achieved");

        prm.declare_entry ("allow recover CFL number", "true",
                           Patterns::Bool(),
                           "allow recover the reduced CFL number when consecutive convenvgence achieved");

        prm.declare_entry ("solution extrapolation length", "1.0",
                           Patterns::Double(),
                           "relative length of the forward extrapolation for predicting solution of next time step");

        prm.declare_entry ("newton linear search length try limit", "0",
                           Patterns::Integer (0,8),
                           "largest index of linear search length could try. "
                           "value of linear search length is defined internally.");

        prm.declare_entry ("iter in stage1", "5",
                           Patterns::Integer (0),
                           "number of interations in stage one for time step increasing in stationary case");
        prm.declare_entry ("step increasing ratio stage1", "1.0",
                           Patterns::Double (0),
                           "step increasing ratio in stage1 one for time step increasing in stationary case");
        prm.declare_entry ("minimum step increasing ratio stage2", "0.1",
                           Patterns::Double (0),
                           "minimum step increasing ratio in stage two for time step increasing in stationary case");
        prm.declare_entry ("step increasing power stage2", "1.2",
                           Patterns::Double (0),
                           "step increasing power in stage two for time step increasing in stationary case");

        prm.declare_entry ("CFL number", "1.0",
                           Patterns::Double (0),
                           "CFL number");
        prm.declare_entry ("unsteady CFL number max", "-1.0",
                           Patterns::Double(),
                           "maximum of CFL number for unsteady simulation");

        prm.declare_entry ("reference time step", "0.1",
                           Patterns::Double(),
                           "simulation time step");
        prm.declare_entry ("final time", "10.0",
                           Patterns::Double (0),
                           "simulation end time");
        prm.declare_entry ("time march tolerance", "-10.0",
                           Patterns::Double(),
                           "Terminate time marching when log10 of error L2 norm is less than this value");
        prm.declare_entry ("theta scheme value", "0.5",
                           Patterns::Double (0,1),
                           "value for theta that interpolated between explicit "
                           "Euler (theta=0), Crank-Nicolson (theta=0.5), and "
                           "implicit Euler (theta=1).");
      }
      prm.leave_subsection();


      for (unsigned int b=0; b<max_n_boundaries; ++b)
        {
          prm.enter_subsection ("boundary_" +
                                Utilities::int_to_string (b));
          {
            prm.declare_entry ("type",
                               "SlipWall",
                               // The default boundary condition type must be "out flow"
                               // to make sure the no penetration boundary condition
                               // working normally. Because we need to extrapolate the density
                               // and energy (pressure) values at the no penetration
                               // boundary just as if there is an outflow boundary.

                               Patterns::Selection ("SlipWall|Symmetry|FarField|PressureOutlet|MomentumInlet|AllPrimitiveValues|MMS_BC"),
                               "<SlipWall|Symmetry|FarField|PressureOutlet|MomentumInlet|AllPrimitiveValues|MMS_BC>");

            for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
              {

                prm.declare_entry ("w_" + Utilities::int_to_string (di) +
                                   " value", "0.0",
                                   Patterns::Anything(),
                                   "expression in x,y,z");
              }
            prm.declare_entry ("integrate force", "false",
                               Patterns::Bool(),
                               "integrate force on this kind of boundary");
          }
          prm.leave_subsection();
        }

      prm.enter_subsection ("initial condition");
      {
        for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
          prm.declare_entry ("w_" + Utilities::int_to_string (di) + " value",
                             "0.0",
                             Patterns::Anything(),
                             "expression in x,y,z");
      }
      prm.leave_subsection();

      Parameters::PhysicalParameters::declare_parameters (prm);
      Parameters::Solver::declare_parameters (prm);
      Parameters::Refinement::declare_parameters (prm);
      Parameters::Flux<dim>::declare_parameters (prm);
      Parameters::Output::declare_parameters (prm);
      Parameters::FEParameters::declare_parameters (prm);
    }


    template <int dim>
    void
    AllParameters<dim>::parse_parameters (ParameterHandler &prm)
    {
      // PhysicalParameters contains lots of critical parameters, do it first.
      Parameters::PhysicalParameters::parse_parameters (prm);

      n_mms = prm.get_integer ("MMS");
      mesh_filename = prm.get ("mesh");
      time_advance_history_filename = prm.get ("time history");
      iteration_history_filename = prm.get ("iteration history");

      {
        const std::string mesh_format_buf = prm.get ("mesh format");
        if (mesh_format_buf == "ucd")
          {
            mesh_format = format_ucd;
          }
        else if (mesh_format_buf == "gmsh")
          {
            mesh_format = format_gmsh;
          }
        else
          {
            AssertThrow (false, ExcNotImplemented());
          }
      }

      scale_mesh = std::abs (prm.get_double ("scale mesh"));

      n_global_refinement = prm.get_integer ("global refinement");

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
      }

      diffusion_power = prm.get_double ("diffusion power");
      diffusion_coefficoent = prm.get_double ("diffusion coefficient");

      gravity = prm.get_double ("gravity");

      prm.enter_subsection ("time stepping");
      {
        rigid_reference_time_step = prm.get_bool ("rigid reference time step");
        CFL_number = prm.get_double ("CFL number");
        CFL_number_max = prm.get_double ("unsteady CFL number max");
        if (CFL_number_max < CFL_number)
          {
            CFL_number_max = CFL_number;
          }

        reference_time_step = prm.get_double ("reference time step");
        if (reference_time_step <= 0)
          {
            is_stationary = true;
            reference_time_step = -reference_time_step;
          }
        else
          {
            is_stationary = false;
          }
        AssertThrow (reference_time_step!=0.0, ExcMessage (" Time step size cann't be 0."));

        auto_CFL_number = prm.get_bool ("auto CFL number");
        allow_recover_CFL_number = prm.get_bool ("allow recover CFL number");

        n_iter_stage1 = prm.get_integer ("iter in stage1");
        step_increasing_ratio_stage1 = prm.get_double ("step increasing ratio stage1");
        minimum_step_increasing_ratio_stage2 = prm.get_double ("minimum step increasing ratio stage2");
        step_increasing_power_stage2 = prm.get_double ("step increasing power stage2");

        final_time = prm.get_double ("final time");
        time_march_tolerance = prm.get_double ("time march tolerance");
        theta = prm.get_double ("theta scheme value");
        solution_extrapolation_length = prm.get_double ("solution extrapolation length");
        newton_linear_search_length_try_limit = prm.get_integer ("newton linear search length try limit");
      }
      prm.leave_subsection();

      for (unsigned int boundary_id=0; boundary_id<max_n_boundaries;
           ++boundary_id)
        {
          prm.enter_subsection ("boundary_" +
                                Utilities::int_to_string (boundary_id));
          {
            std::vector<std::string>
            expressions (EquationComponents<dim>::n_components, "0.0");

            const std::string boundary_type
              = prm.get ("type");

            if (boundary_type == "SlipWall" || boundary_type == "Symmetry")
              {
                boundary_conditions[boundary_id].kind = Boundary::Symmetry;
              }
            else if (boundary_type == "FarField")
              {
                boundary_conditions[boundary_id].kind = Boundary::FarField;
              }
            else if (boundary_type == "PressureOutlet")
              {
                boundary_conditions[boundary_id].kind = Boundary::PressureOutlet;
              }
            else if (boundary_type == "MomentumInlet")
              {
                boundary_conditions[boundary_id].kind = Boundary::MomentumInlet;
              }
            else if (boundary_type == "AllPrimitiveValues")
              {
                boundary_conditions[boundary_id].kind = Boundary::AllPrimitiveValues;
              }
            else if (boundary_type == "MMS_BC")
              {
                boundary_conditions[boundary_id].kind = Boundary::MMS_BC;
              }
            else
              {
                AssertThrow (false, ExcNotImplemented());
              }

            for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
              {
                expressions[di] = prm.get ("w_" + Utilities::int_to_string (di) +
                                           " value");
              }

            boundary_conditions[boundary_id].values
            .initialize (FunctionParser<dim>::default_variable_names(),
                         expressions,
                         std::map<std::string, double>());
            sum_force[boundary_id] = prm.get_bool ("integrate force");
          }
          prm.leave_subsection();
        }

      prm.enter_subsection ("initial condition");
      {
        std::vector<std::string> expressions (EquationComponents<dim>::n_components,
                                              "0.0");
        for (unsigned int di = 0; di < EquationComponents<dim>::n_components; di++)
          expressions[di] = prm.get ("w_" + Utilities::int_to_string (di) +
                                     " value");
        initial_conditions.initialize (FunctionParser<dim>::default_variable_names(),
                                       expressions,
                                       std::map<std::string, double>());
      }
      prm.leave_subsection();

      Parameters::Solver::parse_parameters (prm);
      Parameters::Refinement::parse_parameters (prm);
      Parameters::Flux<dim>::parse_parameters (prm);
      Parameters::Output::parse_parameters (prm);
      Parameters::FEParameters::parse_parameters (prm);
    }

    template struct AllParameters<2>;
    template struct AllParameters<3>;

  } /* End namespace Parameters */

} /* End of namespace NSFEMSolver */
