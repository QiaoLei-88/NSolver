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
    void Flux::declare_parameters (ParameterHandler &prm)
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

    void Flux::parse_parameters (ParameterHandler &prm)
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

      for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
        {
          prm.declare_entry ("diffusion factor for w_" + Utilities::int_to_string (di),
                             "1.0",
                             Patterns::Double (0),
                             "diffusion factor for component w_" + Utilities::int_to_string (di));
        }

      prm.declare_entry ("gravity", "-1.0",
                         Patterns::Double(),
                         "gravity");

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

      Parameters::TimeStepping::declare_parameters (prm);
      Parameters::PhysicalParameters::declare_parameters (prm);
      Parameters::Solver::declare_parameters (prm);
      Parameters::Refinement::declare_parameters (prm);
      Parameters::Flux::declare_parameters (prm);
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

      for (unsigned int di=0; di<EquationComponents<dim>::n_components; ++di)
        {
          diffusion_factor[di] = prm.get_double ("diffusion factor for w_"
                                                 + Utilities::int_to_string (di));
        }

      gravity = prm.get_double ("gravity");

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

      Parameters::TimeStepping::parse_parameters (prm);
      Parameters::Solver::parse_parameters (prm);
      Parameters::Refinement::parse_parameters (prm);
      Parameters::Flux::parse_parameters (prm);
      Parameters::Output::parse_parameters (prm);
      Parameters::FEParameters::parse_parameters (prm);
    }

    template struct AllParameters<2>;
    template struct AllParameters<3>;

  } /* End namespace Parameters */

} /* End of namespace NSFEMSolver */
