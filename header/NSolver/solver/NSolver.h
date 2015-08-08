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

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

// Header files for MPI parallel computing
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>

#define USE_TRILINOS_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}
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

#include <NSolver/Parameters/AllParameters.h>
#include <NSolver/Parameters/FEParameters.h>
#include <NSolver/EquationComponents.h>
#include <NSolver/NSEquation.h>
#include <NSolver/Postprocessor.h>
#include <NSolver/MMS.h>
#include <NSolver/WallForce.h>
#include <NSolver/linearVelocityPotential/linearVelocityPotential.h>


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

#ifndef __NSVector__DEFINED__
  typedef LA::MPI::Vector NSVector;
#define __NSVector__DEFINED__
#endif
  template <int dim>
  class NSolver
  {
  public:
    NSolver (Parameters::AllParameters<dim> *const para_ptr_in);
    void run();

  private:
    void setup_system();
    void initialize();
    void check_negative_density_pressure() const;

    void calc_time_step();
    void assemble_system (const unsigned int nonlin_iter);
    void assemble_cell_term (const FEValues<dim>             &fe_v,
                             const std::vector<types::global_dof_index> &dofs,
                             const unsigned int cell_index,
                             const unsigned int nonlin_iter);
    void assemble_face_term (const unsigned int               face_no,
                             const FEFaceValuesBase<dim>     &fe_v,
                             const FEFaceValuesBase<dim>     &fe_v_neighbor,
                             const std::vector<types::global_dof_index> &dofs,
                             const std::vector<types::global_dof_index> &dofs_neighbor,
                             const bool                       external_face,
                             const unsigned int               boundary_id,
                             const double                     face_diameter);

    std::pair<unsigned int, double> solve (NSVector &solution);

    void integrate_force (Parameters::AllParameters<dim> const *const parameters,
                          WallForce &wall_force) const;

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
    // known that for transsonic simulations with the Euler equations,
    // computations do not converge even as $h\rightarrow 0$ if the boundary
    // approximation is not of sufficiently high order.


    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;

    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    // TimerOutput                               computing_timer;

    const MappingQ1<dim> mapping;

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
    // Cache up the right_hand_side for out put at the first Newton iteration
    // of each time step.


    // All output relevant vectors need to have parallel capability and be
    // size to "locally_relevant_dofs" because ghost cell is needed to
    // determine the outmost face values.
    NSVector      residual_for_output;

    Vector<double>       entropy_viscosity;
    Vector<double>       cellSize_viscosity;
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

    // The following statement is equivalent to
    // TrilinosWrappers::SparseMatrix system_matrix. It is defined by
    // typedef TrilinosWrappers::SparseMatrix SparseMatrix inside namespace
    // MPI in generic_linear_algebra.h
    LA::MPI::SparseMatrix system_matrix;

    const bool I_am_host;
    const unsigned int myid;
    const SmartPointer<Parameters::AllParameters<dim> >    parameters;
    ConditionalOStream              verbose_cout;
    ConditionalOStream              pcout;
    TimerOutput                     computing_timer;

    //Introduce MMS
    MMS mms;
    std_cxx11::array<double, EquationComponents<dim>::n_components> mms_error_l2;
    double mms_error_linfty;

    // Runtime parameters, not specified in input file
    double time_step;
    double CFL_number;
    int n_sparsity_pattern_out;
  };

}
#endif /* defined(__NSolver__NSolver__) */
