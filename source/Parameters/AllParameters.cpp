//
//  AllParameters.cpp
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/Parameters/AllParameters.h>

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
        prm.declare_entry ("Preconditioner", "AZ_DD",
                           Patterns::Selection ("NoPrec|AZ_DD|AZ_AMG|MDFILU"),
                           "Preconditioner for iterative solver.");
        prm.declare_entry ("ILU level", "0",
                           Patterns::Integer (0),
                           "Fill in level of ILU preconditioner");
        prm.declare_entry ("residual", "1e-10",
                           Patterns::Double(),
                           "Linear solver residual");
        prm.declare_entry ("max iters", "300",
                           Patterns::Integer(),
                           "Maximum solver iterations");

        prm.declare_entry ("RCM reorder", "false",
                           Patterns::Bool(),
                           "Do Reverse Cuthill–McKee reordering.");
        prm.declare_entry ("AZ_kspace", "150",
                           Patterns::Integer (0),
                           "Size of Krylov space in GMRES solver");
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
        {
          const std::string buff = prm.get ("Preconditioner");
          if (buff == "NoPrec")
            {
              prec_type = NoPrec;
            }
          else if (buff == "AZ_DD")
            {
              prec_type = AZ_DD;
            }
          else if (buff == "AZ_AMG")
            {
              prec_type = AZ_AMG;
            }
          else if (buff == "MDFILU")
            {
              prec_type = MDFILU;
            }
          else
            {
              AssertThrow (false, ExcNotImplemented());
            }
        }
        ILU_level = prm.get_integer ("ILU level");

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

        // Do not relay on C++ standard to determine value of boolean
        AZ_RCM_reorder  = (prm.get_bool ("RCM reorder")?1:0);
        AZ_Krylov_space = prm.get_integer ("AZ_kspace");
        ilut_fill       = prm.get_double ("ilut fill");
        ilut_atol       = prm.get_double ("ilut absolute tolerance");
        ilut_rtol       = prm.get_double ("ilut relative tolerance");
        ilut_drop       = prm.get_double ("ilut drop tolerance");
      }
      prm.leave_subsection();
    }

//Refinement
    template<int dim>
    Refinement<dim>::Refinement()
      :
      component_mask (EquationComponents<dim>::n_components, true)
    {}

    template<int dim>
    void Refinement<dim>::declare_parameters (ParameterHandler &prm)
    {

      prm.enter_subsection ("refinement");
      {
        prm.declare_entry ("refinement indicator", "Gradient",
                           Patterns::Selection ("Gradient|Kelly"),
                           "type of refinement indicator");
        prm.declare_entry ("refine fraction", "0.1",
                           Patterns::Double(),
                           "Fraction of gird refinement");
        prm.declare_entry ("coarsen fraction", "0.1",
                           Patterns::Double(),
                           "Fraction of grid coarsening");
        prm.declare_entry ("shock value", "4.0",
                           Patterns::Double(),
                           "value for shock indicator");
        prm.declare_entry ("max refine level", "3.0",
                           Patterns::Double(),
                           "number of max refinement levels");
        prm.declare_entry ("max refine time", "0",
                           Patterns::Double(),
                           "stop mesh refinement after this number of step");
        prm.declare_entry ("max cells", "-4.0",
                           Patterns::Double(),
                           "maximum number (positive value) or maximum number ratio (negative value) of cells");
        prm.declare_entry ("max cell size", "-1.0",
                           Patterns::Double(),
                           "maximum cell size in mesh refinement");
        prm.declare_entry ("min cell size", "-1.0",
                           Patterns::Double(),
                           "minimum cell size in mesh refinement");
        prm.declare_entry ("component mask", "65535",
                           Patterns::Integer (0),
                           "Equation components want to use in Kelly error estimator");
      }
      prm.leave_subsection();
    }

    template<int dim>
    void Refinement<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("refinement");
      {
        {
          const std::string buff = prm.get ("refinement indicator");
          if (buff == "Gradient")
            {
              refinement_indicator = Gradient;
            }
          else if (buff == "Kelly")
            {
              refinement_indicator = Kelly;
            }
          else
            {
              AssertThrow (false, ExcNotImplemented());
            }
        }
        refine_fraction  = prm.get_double ("refine fraction");
        coarsen_fraction = prm.get_double ("coarsen fraction");
        max_cells        = prm.get_double ("max cells");
        {
          unsigned const mask_int = prm.get_integer ("component mask");
          for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
            {
              component_mask.set (ic, mask_int & (1<<ic));
            }
        }
        shock_val        = prm.get_double ("shock value");
        max_refine_level = prm.get_double ("max refine level");
        max_refine_time  = prm.get_double ("max refine time");
        if (max_refine_time < 0.0)
          {
            max_refine_time = std::numeric_limits<double>::max();
          }
        max_cell_size    = prm.get_double ("max cell size");
        if (max_cell_size < 0.0)
          {
            max_cell_size = std::numeric_limits<double>::max();
          }
        min_cell_size    = prm.get_double ("min cell size");
        if (min_cell_size < 0.0)
          {
            min_cell_size = std::numeric_limits<double>::min();
          }
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
        prm.declare_entry ("step", "0.0",
                           Patterns::Double (0.0),
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
                         "Number of manufactured solution to test against");
      prm.declare_entry ("MMS use strong BC", "false",
                         Patterns::Bool(),
                         "Use strong boundary for MMS boundary condition");
      prm.declare_entry ("ManifoldCircle", "0",
                         Patterns::Integer(),
                         "Activate C1Circle test case.");
      prm.declare_entry ("NACA_foil", "0",
                         Patterns::Integer(),
                         "Use NACA 4 digit foil as boundary manifold.");

      prm.declare_entry ("mesh", "grid.inp",
                         Patterns::Anything(),
                         "input file name");

      prm.declare_entry ("mesh format", "gmsh",
                         Patterns::Anything(),
                         "input mesh file format");

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

      prm.declare_entry ("renumber dofs", "None",
                         Patterns::Selection ("None|RCM|RCM_WithStartPoint"),
                         "How to renumber dofs");

      prm.declare_entry ("renumber start point x", "0",
                         Patterns::Double(),
                         "x coordinate of renumber start point");
      prm.declare_entry ("renumber start point y", "0",
                         Patterns::Double(),
                         "y coordinate of renumber start point");
      prm.declare_entry ("renumber start point z", "0",
                         Patterns::Double(),
                         "z coordinate of renumber start point");
      prm.declare_entry ("output sparsity pattern", "false",
                         Patterns::Bool(),
                         "output sparsity pattern or not");
      prm.declare_entry ("output system matrix", "false",
                         Patterns::Bool(),
                         "output system matrix or not");

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

                               Patterns::Selection ("SlipWall|Symmetry|FarField|NonSlipWall|PressureOutlet|MomentumInlet|AllPrimitiveValues|MMS_BC"),
                               "<SlipWall|Symmetry|FarField|NonSlipWall|PressureOutlet|MomentumInlet|AllPrimitiveValues|MMS_BC>");

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
        prm.declare_entry ("init method", "UserFunction",
                           Patterns::Selection ("UserFunction|FreeStream|LinearVelocityPotential|FullVelocityPotential"),
                           "<UserFunction|FreeStream|LinearVelocityPotential|FullVelocityPotential>");
        prm.declare_entry ("init fe degree", "1", Patterns::Integer(),
                           "FE degree used for solving velocity potential equation");

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
      Parameters::Refinement<dim>::declare_parameters (prm);
      Parameters::Flux::declare_parameters (prm);
      Parameters::Output::declare_parameters (prm);
      Parameters::FEParameters::declare_parameters (prm);
      Parameters::StabilizationParameters<dim>::declare_parameters (prm);
    }


    template <int dim>
    void
    AllParameters<dim>::parse_parameters (ParameterHandler &prm)
    {
      // PhysicalParameters contains lots of critical parameters, do it first.
      Parameters::PhysicalParameters::parse_parameters (prm);

      n_mms = prm.get_integer ("MMS");
      mms_use_strong_BC = prm.get_bool ("MMS use strong BC");
      manifold_circle = prm.get_integer ("ManifoldCircle");
      NACA_foil = prm.get_integer ("NACA_foil");

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
        const std::string prm_buf = prm.get ("renumber dofs");
        if (prm_buf == "None")
          {
            renumber_dofs = None;
          }
        else if (prm_buf == "RCM")
          {
            renumber_dofs = RCM;
          }
        else if (prm_buf == "RCM_WithStartPoint")
          {
            renumber_dofs = RCM_WithStartPoint;
          }
        else
          {
            AssertThrow (false, ExcNotImplemented());
          }
        renumber_start_point[0] = prm.get_double ("renumber start point x");
        renumber_start_point[1] = prm.get_double ("renumber start point y");
        renumber_start_point[2] = prm.get_double ("renumber start point z");

        output_sparsity_pattern = prm.get_bool ("output sparsity pattern");
        output_system_matrix    = prm.get_bool ("output system matrix");
      }

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
            else if (boundary_type == "NonSlipWall")
              {
                boundary_conditions[boundary_id].kind = Boundary::NonSlipWall;
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
        {
          const std::string prm_buf = prm.get ("init method");
          if (prm_buf == "UserFunction")
            {
              init_method = UserFunction;
            }
          else if (prm_buf == "FreeStream")
            {
              init_method = FreeStream;
            }
          else if (prm_buf == "LinearVelocityPotential")
            {
              init_method = LinearVelocityPotential;
            }
          else if (prm_buf == "FullVelocityPotential")
            {
              init_method = FullVelocityPotential;
            }
          else
            {
              AssertThrow (false, ExcNotImplemented());
            }
        }

        init_fe_degree = prm.get_integer ("init fe degree");
        AssertThrow (init_fe_degree<9, ExcMessage ("Do not use element order higher than 8 for potential equation."));
        if (init_fe_degree < 1)
          {
            init_fe_degree = 1;
          }

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
      Parameters::Refinement<dim>::parse_parameters (prm);
      Parameters::Flux::parse_parameters (prm);
      Parameters::Output::parse_parameters (prm);
      Parameters::FEParameters::parse_parameters (prm);
      Parameters::StabilizationParameters<dim>::parse_parameters (prm);
    }

    template struct Refinement<2>;
    template struct Refinement<3>;

    template struct AllParameters<2>;
    template struct AllParameters<3>;
  } /* End namespace Parameters */

} /* End of namespace NSFEMSolver */
