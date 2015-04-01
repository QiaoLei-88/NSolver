//
//  ConservationLaw.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__ConservationLaw__
#define __NSolver__ConservationLaw__

// First a standard set of deal.II includes. Nothing special to comment on
// here:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

//Include grid_tools to scale mesh.
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>

// Then, as mentioned in the introduction, we use various Trilinos packages as
// linear solvers as well as for automatic differentiation. These are in the
// following include files.
//
// Since deal.II provides interfaces to the basic Trilinos matrices, vectors,
// preconditioners and solvers, we include them similarly as deal.II linear
// algebra structures.
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>


// Sacado is the automatic differentiation package within Trilinos, which is
// used to find the Jacobian for a fully implicit Newton iteration:
#include <Sacado.hpp>


// And this again is C++:
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "AllParameters.h"
// Here finally comes the class that actually does something with all the
// Euler equation and parameter specifics we've defined above. The public
// interface is pretty much the same as always (the constructor now takes
// the name of a file from which to read parameters, which is passed on the
// command line). The private function interface is also pretty similar to
// the usual arrangement, with the <code>assemble_system</code> function
// split into three parts: one that contains the main loop over all cells
// and that then calls the other two for integrals over cells and faces,
// respectively.

namespace Step33
{
  using namespace dealii;

  template <int dim>
  class ConservationLaw
  {
  public:
    ConservationLaw (const char *input_filename);
    void run ();

  private:
    void setup_system ();

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

    std::pair<unsigned int, double> solve (Vector<double> &solution);

    void compute_refinement_indicators (Vector<double> &indicator) const;
    void refine_grid (const Vector<double> &indicator);

    void output_results () const;



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
    Triangulation<dim>   triangulation;
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
    Vector<double>       old_solution;
    Vector<double>       current_solution;
    Vector<double>       current_solution_backup;
    Vector<double>       predictor;

    Vector<double>       right_hand_side;

    Vector<double>       entropy_viscosity;
    Vector<double>       cellSize_viscosity;

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
    TrilinosWrappers::SparseMatrix system_matrix;

    Parameters::AllParameters<dim>  parameters;
    ConditionalOStream              verbose_cout;
  };

}
#endif /* defined(__NSolver__ConservationLaw__) */
