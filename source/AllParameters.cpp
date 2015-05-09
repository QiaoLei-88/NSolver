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
        prm.declare_entry ("stab", "mesh",
                           Patterns::Selection ("constant|mesh"),
                           "Whether to use a constant stabilization parameter or "
                           "a mesh-dependent one");
        prm.declare_entry ("stab value", "1",
                           Patterns::Double(),
                           "alpha stabilization");
      }
      prm.leave_subsection();
    }


    void Flux::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("flux");
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

        stabilization_value = prm.get_double ("stab value");
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
      values (EulerEquations<dim>::n_components)
    {}


    template <int dim>
    AllParameters<dim>::AllParameters()
      :
      initial_conditions (EulerEquations<dim>::n_components)
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
        prm.declare_entry ("rigid time step", "false",
                           Patterns::Bool(),
                           "whether use specified time step or"
                           "calulate according to CFL condition.");

        prm.declare_entry ("allow double time step", "false",
                           Patterns::Bool(),
                           "allow double time step size when consecutive convenvgence achieved");

        prm.declare_entry ("CFL number", "1.0",
                           Patterns::Double (0),
                           "CFL number");
        prm.declare_entry ("time step", "0.1",
                           Patterns::Double (0),
                           "simulation time step");
        prm.declare_entry ("final time", "10.0",
                           Patterns::Double (0),
                           "simulation end time");
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
            prm.declare_entry ("no penetration", "false",
                               Patterns::Bool(),
                               "whether the named boundary allows gas to "
                               "penetrate or is a rigid wall");

            for (unsigned int di=0; di<EulerEquations<dim>::n_components; ++di)
              {
                prm.declare_entry ("w_" + Utilities::int_to_string (di),
                                   "outflow",
                                   // The default boundary condition type must be "out flow"
                                   // to make sure the no penetration boundary condition
                                   // working normally. Because we need to extrapolate the density
                                   // and energy (pressure) values at the no penetration
                                   // boundary just as if there is an outflow boundary.
                                   Patterns::Selection ("inflow|outflow|pressure|Riemann|MMS_BC"),
                                   "<inflow|outflow|pressure|Riemann|MMS_BC>");

                prm.declare_entry ("w_" + Utilities::int_to_string (di) +
                                   " value", "0.0",
                                   Patterns::Anything(),
                                   "expression in x,y,z");
              }
          }
          prm.leave_subsection();
        }

      prm.enter_subsection ("initial condition");
      {
        for (unsigned int di=0; di<EulerEquations<dim>::n_components; ++di)
          prm.declare_entry ("w_" + Utilities::int_to_string (di) + " value",
                             "0.0",
                             Patterns::Anything(),
                             "expression in x,y,z");
      }
      prm.leave_subsection();

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
        is_rigid_timestep_size = prm.get_bool ("rigid time step");
        CFL_number = prm.get_double ("CFL number");
        readin_time_step = prm.get_double ("time step");
        if (readin_time_step == 0)
          {
            is_stationary = true;
            readin_time_step = 1.0;
            final_time = 1.0;
          }
        else
          {
            is_stationary = false;
          }
        allow_double_time_step = prm.get_bool ("allow double time step");

        final_time = prm.get_double ("final time");
        theta = prm.get_double ("theta scheme value");
      }
      prm.leave_subsection();

      for (unsigned int boundary_id=0; boundary_id<max_n_boundaries;
           ++boundary_id)
        {
          prm.enter_subsection ("boundary_" +
                                Utilities::int_to_string (boundary_id));
          {
            std::vector<std::string>
            expressions (EulerEquations<dim>::n_components, "0.0");

            const bool no_penetration = prm.get_bool ("no penetration");

            for (unsigned int di=0; di<EulerEquations<dim>::n_components; ++di)
              {
                const std::string boundary_type
                  = prm.get ("w_" + Utilities::int_to_string (di));

                if ((di < dim) && (no_penetration == true))
                  //"(di<dim)" means no_penetration boundary
                  //condition only effect to momentum components.
                  //Other components (i.e. density and energy) will get
                  //wrong values when the default BC type is not "outflow".
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::no_penetration_boundary;
                else if (boundary_type == "inflow")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::inflow_boundary;
                else if (boundary_type == "pressure")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::pressure_boundary;
                else if (boundary_type == "outflow")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::outflow_boundary;
                else if (boundary_type == "Riemann")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::Riemann_boundary;
                else if (boundary_type == "MMS_BC")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::MMS_BC;
                else
                  {
                    AssertThrow (false, ExcNotImplemented());
                  }

                expressions[di] = prm.get ("w_" + Utilities::int_to_string (di) +
                                           " value");
              }

            boundary_conditions[boundary_id].values
            .initialize (FunctionParser<dim>::default_variable_names(),
                         expressions,
                         std::map<std::string, double>());
          }
          prm.leave_subsection();
        }

      prm.enter_subsection ("initial condition");
      {
        std::vector<std::string> expressions (EulerEquations<dim>::n_components,
                                              "0.0");
        for (unsigned int di = 0; di < EulerEquations<dim>::n_components; di++)
          expressions[di] = prm.get ("w_" + Utilities::int_to_string (di) +
                                     " value");
        initial_conditions.initialize (FunctionParser<dim>::default_variable_names(),
                                       expressions,
                                       std::map<std::string, double>());
      }
      prm.leave_subsection();

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
