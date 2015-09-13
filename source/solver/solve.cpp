//
//  NSolver::solve.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


  // @sect4{NSolver::solve}
  //
  // Here, we actually solve the linear system, using either of Trilinos'
  // Aztec or Amesos linear solvers. The result of the computation will be
  // written into the argument vector passed to this function. The result is a
  // pair of number of iterations and the final linear residual.

  template <int dim>
  std::pair<unsigned int, double>
  NSolver<dim>::solve (NSVector &newton_update)
  {
    switch (parameters->solver)
      {
      // If the parameter file specified that a direct solver shall be used,
      // then we'll get here. The process is straightforward, since deal.II
      // provides a wrapper class to the Amesos direct solver within
      // Trilinos. All we have to do is to create a solver control object
      // (which is just a dummy object here, since we won't perform any
      // iterations), and then create the direct solver object. When
      // actually doing the solve, note that we don't pass a
      // preconditioner. That wouldn't make much sense for a direct solver
      // anyway.  At the end we return the solver control statistics &mdash;
      // which will tell that no iterations have been performed and that the
      // final linear residual is zero, absent any better information that
      // may be provided here:
      case Parameters::Solver::direct:
      {
        SolverControl solver_control (1,0);
        TrilinosWrappers::SolverDirect direct (solver_control,
                                               parameters->output ==
                                               Parameters::Solver::verbose);

        direct.solve (system_matrix, newton_update, right_hand_side);

        return std::pair<unsigned int, double> (solver_control.last_step(),
                                                solver_control.last_value());
      }

      // Likewise, if we are to use an iterative solver, we use Aztec's GMRES
      // solver. We could use the Trilinos wrapper classes for iterative
      // solvers and preconditioners here as well, but we choose to use an
      // Aztec solver directly. For the given problem, Aztec's internal
      // preconditioner implementations are superior over the ones deal.II has
      // wrapper classes to, so we use ILU-T preconditioning within the
      // AztecOO solver and set a bunch of options that can be changed from
      // the parameter file.
      //
      // There are two more practicalities: Since we have built our right hand
      // side and solution vector as deal.II Vector objects (as opposed to the
      // matrix, which is a Trilinos object), we must hand the solvers
      // Trilinos Epetra vectors.  Luckily, they support the concept of a
      // 'view', so we just send in a pointer to our deal.II vectors. We have
      // to provide an Epetra_Map for the vector that sets the parallel
      // distribution, which is just a dummy object in serial. The easiest way
      // is to ask the matrix for its map, and we're going to be ready for
      // matrix-vector products with it.
      //
      // Secondly, the Aztec solver wants us to pass a Trilinos
      // Epetra_CrsMatrix in, not the deal.II wrapper class itself. So we
      // access to the actual Trilinos matrix in the Trilinos wrapper class by
      // the command trilinos_matrix(). Trilinos wants the matrix to be
      // non-constant, so we have to manually remove the constantness using a
      // const_cast.
      case Parameters::Solver::gmres:
      {
        Epetra_Vector x (View, system_matrix.trilinos_matrix().DomainMap(),
                         newton_update.begin());
        Epetra_Vector b (View, system_matrix.trilinos_matrix().RangeMap(),
                         right_hand_side.begin());

        AztecOO solver;
        solver.SetAztecOption (AZ_output,
                               (parameters->output ==
                                Parameters::Solver::quiet
                                ?
                                AZ_none
                                :
                                AZ_all));
        solver.SetAztecOption (AZ_solver, AZ_gmres);
        solver.SetAztecOption (AZ_kspace, parameters->AZ_Krylov_space);
        solver.SetRHS (&b);
        solver.SetLHS (&x);

        if (parameters->prec_type == Parameters::Solver::NoPrec)
          {
            solver.SetAztecOption (AZ_precond,         AZ_none);
          }
        else if (parameters->prec_type == Parameters::Solver::AZ_DD)
          {
            solver.SetAztecOption (AZ_precond,         AZ_dom_decomp);
            solver.SetAztecOption (AZ_subdomain_solve, AZ_ilut);
            solver.SetAztecOption (AZ_overlap,         0);
            solver.SetAztecOption (AZ_reorder,  parameters->AZ_RCM_reorder);

            solver.SetAztecParam (AZ_drop,      parameters->ilut_drop);
            solver.SetAztecParam (AZ_ilut_fill, parameters->ilut_fill);
            solver.SetAztecParam (AZ_athresh,   parameters->ilut_atol);
            solver.SetAztecParam (AZ_rthresh,   parameters->ilut_rtol);
          }

        solver.SetUserMatrix (const_cast<Epetra_CrsMatrix *>
                              (&system_matrix.trilinos_matrix()));

        switch (parameters->prec_type)
          {
          case Parameters::Solver::MDFILU:
          {
            std::cerr << "Initialize MDFILU\n";
            const unsigned estimated_row_length
              = 2 * system_matrix.n_nonzero_elements()/system_matrix.m();
            std::cerr << "n_nonzero_elements: " << system_matrix.n_nonzero_elements()
                      << std::endl;
            std::cerr << "system_matrix.m(): " << system_matrix.m()
                      << std::endl;
            MDFILU mdfilu (system_matrix,
                           estimated_row_length,
                           parameters->ILU_level);
            std::cerr << "number of new fill in: " << mdfilu.number_of_new_fill_ins()
                      << std::endl;
            mdfilu.SetUseTranspose (false);
            solver.SetPrecOperator (&mdfilu);
            std::cerr << "Start Iterate\n";
            solver.Iterate (parameters->max_iterations, parameters->linear_residual);
            break;
          }
          case Parameters::Solver::AZ_DD:
          {
            solver.Iterate (parameters->max_iterations, parameters->linear_residual);
            break;
          }
          default:
          {
            AssertThrow (false, ExcNotImplemented());
            break;
          }
          }

        return std::pair<unsigned int, double> (solver.NumIters(),
                                                solver.TrueResidual());
      }
      }

    Assert (false, ExcNotImplemented());
    return std::pair<unsigned int, double> (0,0);
  }

#include "NSolver.inst"
}
