//
//  NSolver.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__NSolver__
#define __NSolver__NSolver__

// First a standard set of deal.II includes. Nothing special to comment on
// here:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/std_cxx11/array.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

//Include grid_tools to scale mesh.
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_c1.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

// Header files for MPI parallel computing
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
// End Header files for MPI

// Then, as mentioned in the introduction, we use various Trilinos packages as
// linear solvers as well as for automatic differentiation. These are in the
// following include files.
//
// Since deal.II provides interfaces to the basic Trilinos matrices,
// preconditioners and solvers, we include them similarly as deal.II linear
// algebra structures.
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
// For Trilinos multi-grid preconditioner
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <ml_MultiLevelPreconditioner.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS
#include <deal.II/lac/trilinos_solver.h>

//MMS
#include <deal.II/grid/grid_generator.h>

// Sacado is the automatic differentiation package within Trilinos, which is
// used to find the Jacobian for a fully implicit Newton iteration:
// It is known that Trilinos::Sacado(at least until version 11.10.2) package
// will trigger a warning.
// Since we are not responsible for this, just suppress the warning by
// the following macro directive.
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <Sacado.hpp>
// Recover diagnostic checks for the rest of code.
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

// And this again is C++:
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <memory>

#include <NSolver/types.h>
#include <NSolver/Parameters/AllParameters.h>
#include <NSolver/Parameters/FEParameters.h>
#include <NSolver/EquationComponents.h>
#include <NSolver/NSEquation.h>
#include <NSolver/Postprocessor.h>
#include <NSolver/MMS.h>
#include <NSolver/WallForce.h>
#include <NSolver/velocityPotential/linearVelocityPotential.h>
#include <NSolver/velocityPotential/fullVelocityPotential.h>
#include <NSolver/MDFILU/MDFILU.h>
#include <NSolver/Tools.h>
#include <NSolver/BoundaryManifold/BndNaca4DigitSymm.h>
#include <NSolver/CellDataTransfer.h>

// Here finally comes the class that actually does something with all the
// Euler equation and parameter specifics we've defined above. The public
// interface is pretty much the same as always (the constructor now takes
// the name of a file from which to read parameters, which is passed on the
// command line). The private function interface is also pretty similar to
// the usual arrangement, with the <code>assemble_system</code> function
// split into three parts: one that contains the main loop over all cells
// and that then calls the other two for integrals over cells and faces,
// respectively.

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  class NSolver
  {
  public:
    NSolver (Parameters::AllParameters<dim> *const para_ptr_in);
    ~NSolver();
    void run();

  private:
    void setup_system();
    void initialize();
    void check_negative_density_pressure() const;

    void calc_time_step();
    void calc_artificial_viscosity();
    void assemble_system();
    void assemble_cell_term (const FEValues<dim>             &fe_v,
                             const std::vector<types::global_dof_index> &dofs);
    /**
     * This function set proper value for laplacian_indicator and laplacian_threshold.
     */
    void calc_laplacian_indicator();
    /**
     * Compute infinity norm of solution and/or gradient jumps across all faces in &p cell.
     */
    double calc_jumps (const FEFaceValuesBase<dim>                          &fe_v_face_this,
                       const FEFaceValuesBase<dim>                          &fe_v_face_neighbor,
                       const unsigned int                                    boundary_id,
                       const bool                                            accumulate_grad_jump,
                       const bool                                            accumulate_value_jump);

    /**
     * Apply Laplacian continuation to system matrix and residual. The coefficients
     * are depending to system residual and solution jump. So this procedure can
     * not be embedded into assemble_system().
     */
    void apply_laplacian_continuation();
    void assemble_face_term (const unsigned int               face_no,
                             const FEFaceValuesBase<dim>     &fe_v,
                             const FEFaceValuesBase<dim>     &fe_v_neighbor,
                             const std::vector<types::global_dof_index> &dofs,
                             const std::vector<types::global_dof_index> &dofs_neighbor,
                             const bool                       external_face,
                             const unsigned int               boundary_id,
                             const double                     face_diameter);
    void apply_strong_boundary_condtions();

    std::pair<unsigned int, double> solve (NSVector &solution,
                                           const double absolute_linear_tolerance);

    void integrate_force (WallForce &wall_force) const;

    void compute_refinement_indicators();
    void refine_grid();

    void output_results() const;



    // The first few member variables are also rather standard. Note that we
    // define a mapping object to be used throughout the program when
    // assembling terms (we will hand it to every FEValues and FEFaceValues
    // object); the mapping we use is just the standard $Q_1$ mapping --
    // nothing fancy, in other words -- but declaring one here and using it
    // throughout the program will make it simpler later on to change it if
    // that should become necessary. This is, in fact, rather pertinent: it is
    // known that for transonic simulations with the Euler equations,
    // computations do not converge even as $h\rightarrow 0$ if the boundary
    // approximation is not of sufficiently high order.


    MPI_Comm                                  mpi_communicator;
    // Must appear before Triangulation to make it initialized before triangulation
    // and destroyed after triangulation.
    const HyperBallBoundary<dim> *spherical_boundary;
    const StraightBoundary<dim> straight_boundary;
    const BndNaca4DigitSymm<dim> NACA_foil_boundary;

    parallel::distributed::Triangulation<dim> triangulation;

    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    // TimerOutput                               computing_timer;

    const Mapping<dim>   *mapping_ptr;

    const FESystem<dim>  fe;
    DoFHandler<dim>      dof_handler;

    const QGauss<dim>    quadrature;
    const QGauss<dim-1>  face_quadrature;

    // Next come a number of data vectors that correspond to the solution of
    // the previous time step (<code>old_solution</code>), the best guess of
    // the current solution (<code>current_solution</code>; we say
    // <i>guess</i> because the Newton iteration to compute it may not have
    // converged yet, whereas <code>old_solution</code> refers to the fully
    // converged final result of the previous time step), and a predictor for
    // the solution at the next time step, computed by extrapolating the
    // current and previous solution one time step into the future:

    NSVector       newton_update;
    NSVector       locally_owned_solution;

    NSVector       current_solution;
    NSVector       old_solution;
    NSVector       old_old_solution;
    NSVector       predictor;

    NSVector       right_hand_side;
    NSVector       physical_residual;
    // Cache up the right_hand_side for out put at the first Newton iteration
    // of each time step.


    // All output relevant vectors need to have parallel capability and be
    // size to "locally_relevant_dofs" because ghost cell is needed to
    // determine the out-most face values.
    NSVector      residual_for_output;

    Vector<double>       local_time_step_size;
    Vector<double>       artificial_viscosity;
    Vector<double>       unsmoothed_artificial_viscosity;
    Vector<double>       old_artificial_viscosity;
    bool                 blend_artificial_viscosity;
    double               mean_artificial_viscosity;
    Vector<double>       artificial_thermal_conductivity;
    Vector<float>        dominant_viscosity;
    Vector<float>        refinement_indicators;

    // This final set of member variables (except for the object holding all
    // run-time parameters at the very bottom and a screen output stream that
    // only prints something if verbose output has been requested) deals with
    // the interface we have in this program to the Trilinos library that
    // provides us with linear solvers. Similarly to including PETSc matrices
    // in step-17, step-18, and step-19, all we need to do is to create a
    // Trilinos sparse matrix instead of the standard deal.II class. The
    // system matrix is used for the Jacobian in each Newton step. Since we do
    // not intend to run this program in parallel (which wouldn't be too hard
    // with Trilinos data structures, though), we don't have to think about
    // anything else like distributing the degrees of freedom.

    // NSMatrix is defined in <NSolver/types.h>
    NSMatrix system_matrix;

    const bool I_am_host;
    const unsigned int myid;
    const SmartPointer<Parameters::AllParameters<dim> const>  parameters;
    /**
     * pointer to the same object to @p parameters but could modify its value.
     */
    Parameters::AllParameters<dim> *const                     parameters_modifier;
    ConditionalOStream              verbose_cout;
    ConditionalOStream              pcout;
    std::ofstream                   paper_data_std;
    ConditionalOStream              paper_data_out;
    TimerOutput                     computing_timer;

    //Introduce MMS
    MMS<dim> mms;
    typename MMS<dim>::F_V mms_error_l2;
    typename MMS<dim>::F_V mms_error_H1_semi;
    double mms_error_linfty;

    // Runtime parameters, not specified in input file
    double global_time_step_size;
    double CFL_number;
    int n_sparsity_pattern_out;
    mutable unsigned int field_output_counter;
    // iteration counter
    unsigned int n_time_step;
    unsigned int n_total_iter;
    unsigned int nonlin_iter;

    // continuation_coefficient
    double continuation_coefficient;
    Vector<double> laplacian_indicator;
    double laplacian_threshold;

    double continuation_coeff_time;
    double continuation_coeff_laplacian;

    double global_Mach_max;
  };

}
#endif /* defined(__NSolver__NSolver__) */
