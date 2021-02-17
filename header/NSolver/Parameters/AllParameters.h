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

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/fe/component_mask.h>

#include <NSolver/BoundaryType.h>
#include <NSolver/EquationComponents.h>
#include <NSolver/NumericalFlux.h>
#include <NSolver/Parameters/FEParameters.h>
#include <NSolver/Parameters/PhysicalParameters.h>
#include <NSolver/Parameters/StabilizationParameters.h>
#include <NSolver/Parameters/TimeSteppingParameters.h>

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
      enum SolverType
      {
        gmres,
        direct
      };
      SolverType solver;

      enum OutputType
      {
        quiet,
        verbose
      };
      OutputType output;

      enum Preconditioner
      {
        NoPrec,
        AZ_DD,
        AZ_AMG,
        MDFILU
      };
      Preconditioner prec_type;
      int            ILU_level;

      double linear_residual;
      int    max_iterations;

      int    AZ_RCM_reorder;
      int    AZ_Krylov_space;
      double ilut_fill;
      double ilut_atol;
      double ilut_rtol;
      double ilut_drop;

      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };

    // @sect4{Parameters::Refinement}
    //
    // Similarly, here are a few parameters that determine how the mesh is to
    // be refined (and if it is to be refined at all). For what exactly the
    // shock parameters do, see the mesh refinement functions further down.
    template <int dim>
    struct Refinement
    {
      enum Indicator
      {
        Gradient,
        Kelly,
        ErrorAndJump
      };

      Indicator refinement_indicator;

      bool   do_refine_on_initial_field;
      double shock_val;
      double refine_fraction;
      double coarsen_fraction;

      /**
       * <code>max_cells</code> upper limit of number of cells during mesh
       * adaptation. <code>ratio_max_cells</code> multiplier on initial mesh
       * number to calculate the cell number limit. The lower value resulted
       * from the two parameters will take effect. Note that the initial mesh
       * can not be coarsen. Incompatible input value of max_cells will be
       * ignored without warning. By default, number of cell can increase to
       * four times of initial mesh.
       */
      double       ratio_max_cells;
      unsigned int max_cells;

      /**
       * maximun refinement level during mesh adaptive.
       */
      int max_refine_level;

      /**
       * Stop mesh refinement after max_refine_time. The limit is compared
       * against n_time_step in steady case and against time in unsteady case.
       * The default value is zero, in which case no mesh adaptation will
       * happen.
       *
       * Negative input value will disable this limit.
       */
      double max_refine_time;
      /**
       * when refinement flag is true, mesh cell with size larger than
       * max_cell_size will be forced to refine. The relevant cell size is
       * obtained by TriaAccessor::minimum_vertex_distance()
       *
       * Negative input value will disable this limit.
       * @warning Zero input value will not disable this limit, which will turn
       * every adaptive refinement into global refinement.
       */
      double max_cell_size;
      /**
       * when refinement flag is true, mesh cell with size smaller than
       * min_cell_size will never be refined. The relevant cell size is obtained
       * by TriaAccessor::minimum_vertex_distance()
       *
       * Negative input value will disable this limit.
       * @note Zero value will not be escaped. But this doesn't hurt in normal cases.
       */
      double min_cell_size;

      ComponentMask component_mask;

      Refinement();
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
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
    struct Flux
    {
      enum StabilizationKind
      {
        constant,
        mesh_dependent
      };
      StabilizationKind stabilization_kind;

      NumericalFlux::Type numerical_flux_type;
      NumericalFlux::Type flux_type_switch_to;
      double              stabilization_value;
      double              tolerance_to_switch_flux;

      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };

    // @sect4{Parameters::Output}
    //
    // Then a section on output parameters. We offer to produce Schlieren
    // plots (the squared gradient of the density, a tool to visualize shock
    // fronts), and a time interval between graphical output in case we don't
    // want an output file every time step.
    struct Output
    {
      bool   schlieren_plot;
      double output_step;

      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
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
    struct AllParameters : public PhysicalParameters,
                           public TimeStepping,
                           public Solver,
                           public Refinement<dim>,
                           public Flux,
                           public Output,
                           public FEParameters,
                           public StabilizationParameters<dim>,
                           public Subscriptor
    {
      static const unsigned int max_n_boundaries = 10;

      struct BoundaryConditions
      {
        typename Boundary::Type kind;
        FunctionParser<dim>     values;

        BoundaryConditions();
      };


      AllParameters();

      std::string mesh_filename;
      std::string time_advance_history_filename;
      std::string iteration_history_filename;

      enum MeshFormat
      {
        format_gmsh,
        format_ucd
      };
      MeshFormat mesh_format;
      double     scale_mesh;
      int        n_global_refinement;

      enum RenumberDofs
      {
        None,
        RCM,
        RCM_WithStartPoint
      };
      RenumberDofs renumber_dofs;
      Point<3>     renumber_start_point;
      bool         output_sparsity_pattern;
      bool         output_system_matrix;

      enum InitMethod
      {
        UserFunction,
        FreeStream,
        LinearVelocityPotential,
        FullVelocityPotential
      };
      InitMethod init_method;

      /**
       * FE degree used for solving velocity potential equation.
       * Degree less than 1 will be reset to 1 without warning;
       * Degree larger than 8 will cause an exception and abort the program.
       **/
      int                 init_fe_degree;
      FunctionParser<dim> initial_conditions;

      BoundaryConditions boundary_conditions[max_n_boundaries];
      bool               sum_force[max_n_boundaries];

      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);

      int  n_mms;
      bool mms_use_strong_BC;
      int  manifold_circle;
      int  NACA_foil;
      int  NACA_cheating_refinement;
    };

  } /* End of namespace Parameters */

} /* End of namespace NSFEMSolver */

#endif /* defined(__NSolver__AllParameters__) */
