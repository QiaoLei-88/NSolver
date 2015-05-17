//
//  AllParameters.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__AllParameters__
#define __NSolver__AllParameters__

// Next a section on flux modifications to make it more stable. In
// particular, two options are offered to stabilize the Lax-Friedrichs
// flux: either choose $\mathbf{H}(\mathbf{a},\mathbf{b},\mathbf{n}) =
// \frac{1}{2}(\mathbf{F}(\mathbf{a})\cdot \mathbf{n} +
// \mathbf{F}(\mathbf{b})\cdot \mathbf{n} + \alpha (\mathbf{a} -
// \mathbf{b}))$ where $\alpha$ is either a fixed number specified in the
// input file, or where $\alpha$ is a mesh dependent value. In the latter
// case, it is chosen as $\frac{h}{2\delta T}$ with $h$ the diameter of
// the face to which the flux is applied, and $\delta T$ the current time
// step.

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include "FEParameters.h"
#include "PhysicalParameters.h"
#include "BoundaryType.h"
#include "EquationComponents.h"
#include "NumericalFlux.h"


namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    // @sect4{Parameters::Solver}
    //
    // The first of these classes deals with parameters for the linear inner
    // solver. It offers parameters that indicate which solver to use (GMRES
    // as a solver for general non-symmetric indefinite systems, or a sparse
    // direct solver), the amount of output to be produced, as well as various
    // parameters that tweak the thresholded incomplete LU decomposition
    // (ILUT) that we use as a preconditioner for GMRES.
    //
    // In particular, the ILUT takes the following parameters:
    // - ilut_fill: the number of extra entries to add when forming the ILU
    //   decomposition
    // - ilut_atol, ilut_rtol: When forming the preconditioner, for certain
    //   problems bad conditioning (or just bad luck) can cause the
    //   preconditioner to be very poorly conditioned.  Hence it can help to
    //   add diagonal perturbations to the original matrix and form the
    //   preconditioner for this slightly better matrix.  ATOL is an absolute
    //   perturbation that is added to the diagonal before forming the prec,
    //   and RTOL is a scaling factor $rtol \geq 1$.
    // - ilut_drop: The ILUT will drop any values that have magnitude less
    //   than this value.  This is a way to manage the amount of memory used
    //   by this preconditioner.
    //
    // The meaning of each parameter is also briefly described in the third
    // argument of the ParameterHandler::declare_entry call in
    // <code>declare_parameters()</code>.
    struct Solver
    {
      enum SolverType { gmres, direct };
      SolverType solver;

      enum  OutputType { quiet, verbose };
      OutputType output;

      double linear_residual;
      int max_iterations;

      int AZ_RCM_reorder;
      double ilut_fill;
      double ilut_atol;
      double ilut_rtol;
      double ilut_drop;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };

    // @sect4{Parameters::Refinement}
    //
    // Similarly, here are a few parameters that determine how the mesh is to
    // be refined (and if it is to be refined at all). For what exactly the
    // shock parameters do, see the mesh refinement functions further down.
    struct Refinement
    {
      bool do_refine;
      double shock_val;
      double shock_levels;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };

    // @sect4{Parameters::Flux}
    //
    // Next a section on flux modifications to make it more stable. In
    // particular, two options are offered to stabilize the Lax-Friedrichs
    // flux: either choose $\mathbf{H}(\mathbf{a},\mathbf{b},\mathbf{n}) =
    // \frac{1}{2}(\mathbf{F}(\mathbf{a})\cdot \mathbf{n} +
    // \mathbf{F}(\mathbf{b})\cdot \mathbf{n} + \alpha (\mathbf{a} -
    // \mathbf{b}))$ where $\alpha$ is either a fixed number specified in the
    // input file, or where $\alpha$ is a mesh dependent value. In the latter
    // case, it is chosen as $\frac{h}{2\delta T}$ with $h$ the diameter of
    // the face to which the flux is applied, and $\delta T$ the current time
    // step.
    template<int dim>
    struct Flux
    {
      enum StabilizationKind { constant, mesh_dependent };
      StabilizationKind stabilization_kind;

      NumericalFlux::Type numerical_flux_type;
      NumericalFlux::Type flux_type_switch_to;
      double stabilization_value;
      double tolerance_to_switch_flux;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };

    // @sect4{Parameters::Output}
    //
    // Then a section on output parameters. We offer to produce Schlieren
    // plots (the squared gradient of the density, a tool to visualize shock
    // fronts), and a time interval between graphical output in case we don't
    // want an output file every time step.
    struct Output
    {
      bool schlieren_plot;
      double output_step;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };

    // @sect4{Parameters::AllParameters}
    //
    // Finally the class that brings it all together. It declares a number of
    // parameters itself, mostly ones at the top level of the parameter file
    // as well as several in section too small to warrant their own
    // classes. It also contains everything that is actually space dimension
    // dependent, like initial or boundary conditions.
    //
    // Since this class is derived from all the ones above, the
    // <code>declare_parameters()</code> and <code>parse_parameters()</code>
    // functions call the respective functions of the base classes as well.
    //
    // Note that this class also handles the declaration of initial and
    // boundary conditions specified in the input file. To this end, in both
    // cases, there are entries like "w_0 value" which represent an expression
    // in terms of $x,y,z$ that describe the initial or boundary condition as
    // a formula that will later be parsed by the FunctionParser
    // class. Similar expressions exist for "w_1", "w_2", etc, denoting the
    // <code>dim+2</code> conserved variables of the Euler system. Similarly,
    // we allow up to <code>max_n_boundaries</code> boundary indicators to be
    // used in the input file, and each of these boundary indicators can be
    // associated with an inflow, outflow, or pressure boundary condition,
    // with homogeneous boundary conditions being specified for each
    // component and each boundary indicator separately.
    //
    // The data structure used to store the boundary indicators is a bit
    // complicated. It is an array of <code>max_n_boundaries</code> elements
    // indicating the range of boundary indicators that will be accepted. For
    // each entry in this array, we store a pair of data in the
    // <code>BoundaryCondition</code> structure: first, an array of size
    // <code>n_components</code> that for each component of the solution
    // vector indicates whether it is an inflow, outflow, or other kind of
    // boundary, and second a FunctionParser object that describes all
    // components of the solution vector for this boundary id at once.
    //
    // The <code>BoundaryCondition</code> structure requires a constructor
    // since we need to tell the function parser object at construction time
    // how many vector components it is to describe. This initialization can
    // therefore not wait till we actually set the formulas the FunctionParser
    // object represents later in
    // <code>AllParameters::parse_parameters()</code>
    //
    // For the same reason of having to tell Function objects their vector
    // size at construction time, we have to have a constructor of the
    // <code>AllParameters</code> class that at least initializes the other
    // FunctionParser object, i.e. the one describing initial conditions.
    template <int dim>
    struct AllParameters
      :
      public PhysicalParameters,
      public Solver,
      public Refinement,
      public Flux<dim>,
      public Output,
      public FEParameters
    {
      static const unsigned int max_n_boundaries = 10;

      struct BoundaryConditions
      {
        typename Boundary::Type kind;
        FunctionParser<dim> values;

        BoundaryConditions();
      };


      AllParameters();

      double diffusion_power;

      double final_time;
      double time_march_tolerance;
      double CFL_number;
      double CFL_number_max;
      double CFL_number_min;
      double reference_time_step;
      double theta;
      double gravity;
      bool is_stationary;

      // if <code>rigid_reference_time_step</code> is ture, reference time step size
      // is set to the value specified in the input file, otherwise the reference
      // time step size is calculated according to CFL condition.
      // This flag is set to true by default.
      bool rigid_reference_time_step;

      // If <code>auto_CFL_number</code> is ture, for unsteady simulation the
      // time step size will be halfed while Newton iteration diverged; for
      // steady simutation, CFL number will be calculated according to Newton
      // residual.
      // If <code>auto_CFL_number</code> is false, the solver will insist on the
      // CFL number specified in input file.
      // This flag is set to true by default.
      bool auto_CFL_number;

      // In unsteady simulation, Solver will decrease CFL number after linear
      // solver diverged, this option allow recover the time step size to the
      // original specified value.
      bool allow_recover_CFL_number;

      // Predict solution of next time step by making a linear extrapolation from current
      // and last time step. This parameter controls the relative length of the
      // forward extrapolation. Specifically,
      // predicted_solution =  current_solution * (1+solution_extrapolation_length)
      //                      -old_solution * solution_extrapolation_length;
      bool solution_extrapolation_length;

      int newton_linear_search_length_try_limit;

      unsigned int n_iter_stage1;
      double step_increasing_ratio_stage1;
      double minimum_step_increasing_ratio_stage2;
      double step_increasing_power_stage2;

      std::string mesh_filename;
      std::string time_advance_history_filename;
      std::string iteration_history_filename;

      enum MeshFormat
      {
        format_gmsh,
        format_ucd
      };
      MeshFormat mesh_format;
      double scale_mesh;
      int n_global_refinement;

      enum DiffusionType
      {
        diffu_entropy,
        diffu_cell_size,
        diffu_const
      };
      DiffusionType diffusion_type;
      double diffusion_coefficoent;

      FunctionParser<dim> initial_conditions;
      BoundaryConditions  boundary_conditions[max_n_boundaries];
      bool sum_force[max_n_boundaries];

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);

      int n_mms;
    };

  } /* End of namespace Parameters */

} /* End of namespace NSFEMSolver */

#endif /* defined(__NSolver__AllParameters__) */
