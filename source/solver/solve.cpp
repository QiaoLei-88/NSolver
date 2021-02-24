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
  NSolver<dim>::solve(NSVector &   newton_update,
                      const double absolute_linear_tolerance)
  {
    std::pair<unsigned int, double> return_value(-1, -1);
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
            SolverControl solver_control(1, 0);
#ifdef USE_PETSC_LA
            PETScWrappers::SparseDirectMUMPS direct(solver_control,
                                                    mpi_communicator);
            direct.set_symmetric_mode(false);
#else
            TrilinosWrappers::SolverDirect::AdditionalData solver_data(
              parameters->output == Parameters::Solver::verbose);
            TrilinosWrappers::SolverDirect direct(solver_control, solver_data);
#endif
            direct.solve(system_matrix, newton_update, right_hand_side);

            return_value.first  = solver_control.last_step();
            return_value.second = solver_control.last_value();
            break;
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
#ifdef USE_PETSC_LA
            SolverControl solver_control(parameters->max_iterations,
                                         absolute_linear_tolerance);
            PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);

            PETScWrappers::PreconditionBlockJacobi preconditioner(
              system_matrix);

            solver.solve(system_matrix,
                         newton_update,
                         right_hand_side,
                         preconditioner);

            return_value.first  = solver_control.last_step();
            return_value.second = solver_control.last_value();
#else
            Epetra_Vector                  x(View,
                            system_matrix.trilinos_matrix().DomainMap(),
                            newton_update.begin());
            Epetra_Vector                  b(View,
                            system_matrix.trilinos_matrix().RangeMap(),
                            right_hand_side.begin());

            AztecOO solver;
            solver.SetAztecOption(
              AZ_output,
              (parameters->output == Parameters::Solver::quiet ? AZ_none :
                                                                 AZ_all));
            solver.SetAztecOption(AZ_solver, AZ_gmres);
            solver.SetAztecOption(AZ_kspace, parameters->AZ_Krylov_space);
            solver.SetRHS(&b);
            solver.SetLHS(&x);

            Epetra_Operator *preconditioner_ptr(0);
            switch (parameters->prec_type)
              {
                case Parameters::Solver::NoPrec:
                  {
                    solver.SetAztecOption(AZ_precond, AZ_none);
                    break;
                  }
                case Parameters::Solver::AZ_DD:
                  {
                    solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
                    solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
                    solver.SetAztecOption(AZ_overlap, 0);
                    solver.SetAztecOption(AZ_reorder,
                                          parameters->AZ_RCM_reorder);

                    solver.SetAztecParam(AZ_drop, parameters->ilut_drop);
                    solver.SetAztecParam(AZ_ilut_fill, parameters->ilut_fill);
                    solver.SetAztecParam(AZ_athresh, parameters->ilut_atol);
                    solver.SetAztecParam(AZ_rthresh, parameters->ilut_rtol);
                    break;
                  }
                case Parameters::Solver::AZ_AMG:
                  {
                    // for parameter_list.set ("null space: dimension", int)
                    const int n_components =
                      EquationComponents<dim>::n_components;
                    // Build the AMG preconditioner.
                    Teuchos::ParameterList parameter_list;
                    // Equation is not elliptic
                    ML_Epetra::SetDefaults("NSSA", parameter_list);
                    parameter_list.set("aggregation: type", "Uncoupled");
                    parameter_list.set("aggregation: block scaling", true);

                    parameter_list.set("smoother: type", "Chebyshev");
                    parameter_list.set("coarse: type", "Amesos-KLU");

                    parameter_list.set("smoother: sweeps", 2);
                    parameter_list.set("cycle applications", 1);
                    // W-cycle
                    // parameter_list.set ("prec type", "MGW");
                    // V-cycle
                    parameter_list.set("prec type", "MGV");

                    parameter_list.set("smoother: Chebyshev alpha", 10.);
                    parameter_list.set("smoother: ifpack overlap", 0);
                    parameter_list.set("aggregation: threshold", 1.0e-4);
                    parameter_list.set("coarse: max size", 2000);
                    // No detail output
                    parameter_list.set("ML output", 0);

                    parameter_list.set("null space: type", "pre-computed");
                    parameter_list.set("null space: dimension", n_components);

                    // Setup constant modes
                    const Epetra_CrsMatrix &matrix =
                      system_matrix.trilinos_matrix();
                    const Epetra_Map &domain_map = matrix.OperatorDomainMap();

                    Epetra_MultiVector distributed_constant_modes(domain_map,
                                                                  n_components);
                    std::vector<TrilinosScalar> dummy(n_components);


                    const dealii::types::global_dof_index my_size =
                      system_matrix.local_size();
                    if (my_size > 0)
                      {
                        for (int ic = 0; ic < n_components; ++ic)
                          {
                            TrilinosScalar *const begin =
                              &(distributed_constant_modes[ic][0]);
                            TrilinosScalar *const end = begin + my_size;
                            std::fill(begin, end, 0);
                          }

                        // here we assume constant FE among all cells
                        const unsigned int dofs_per_cell = fe.dofs_per_cell;
                        std::vector<dealii::types::global_dof_index>
                          dof_indices(dofs_per_cell);

                        typename DoFHandler<dim>::active_cell_iterator cell =
                          dof_handler.begin_active();
                        const typename DoFHandler<dim>::active_cell_iterator
                          endc = dof_handler.end();
                        for (; cell != endc; ++cell)
                          if (cell->is_locally_owned())
                            {
                              cell->get_dof_indices(dof_indices);
                              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                  const unsigned int component_index =
                                    cell->get_fe()
                                      .system_to_component_index(i)
                                      .first;
                                  const long long int global_dof_index =
                                    dof_indices[i];
                                  const int local_dof_index =
                                    domain_map.LID(global_dof_index);
                                  if (local_dof_index != -1)
                                    {
                                      // For locally owned DoFs
                                      distributed_constant_modes
                                        [component_index][local_dof_index] = 1;
                                    }
                                }
                            }
                        parameter_list.set("null space: vectors",
                                           distributed_constant_modes.Values());
                      }
                    else
                      {
                        // We need to set a valid pointer to data even if there
                        // is no data on the current processor. Therefore, pass
                        // a dummy in that case
                        parameter_list.set("null space: vectors", &dummy[0]);
                      }
                    preconditioner_ptr =
                      new ML_Epetra::MultiLevelPreconditioner(matrix,
                                                              parameter_list);
                    Assert(preconditioner_ptr,
                           ExcMessage("Preconditioner setup failed."));
                    break;
                  }
                case Parameters::Solver::MDFILU:
                  {
                    std::cerr << "Initialize MDFILU\n";
                    const unsigned estimated_row_length =
                      2 * system_matrix.n_nonzero_elements() /
                      system_matrix.m();
                    std::cerr << "n_nonzero_elements: "
                              << system_matrix.n_nonzero_elements()
                              << std::endl;
                    std::cerr << "system_matrix.m(): " << system_matrix.m()
                              << std::endl;
                    preconditioner_ptr = new MDFILU(system_matrix,
                                                    estimated_row_length,
                                                    parameters->ILU_level);
                    // Casting pointer type is not recommended, however here it
                    // is just use for debug output.
                    std::cerr << "number of new fill in: "
                              << static_cast<MDFILU *>(preconditioner_ptr)
                                   ->number_of_new_fill_ins()
                              << std::endl;
                    preconditioner_ptr->SetUseTranspose(false);
                    std::cerr << "Start Iterate\n";
                    break;
                  }
                default:
                  {
                    AssertThrow(false,
                                ExcMessage(
                                  "Preconditioner type not implemented."));
                    break;
                  }
              } // End of switch (parameters->prec_type)

            solver.SetUserMatrix(
              const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
            // SetUserMatrix will set up an internal preconditioner, switch
            // to user defined preconditioner when available.
            if (preconditioner_ptr)
              {
                solver.SetPrecOperator(preconditioner_ptr);
              }
            solver.Iterate(parameters->max_iterations,
                           absolute_linear_tolerance);

            return_value.first  = solver.NumIters();
            return_value.second = solver.TrueResidual();

            // Safety of deleting NULL pointer is assured by C++ standard
            delete preconditioner_ptr;
            preconditioner_ptr = 0;
#endif
            break;
            // End case Parameters::Solver::gmres:
          }
        default:
          {
            AssertThrow(false, ExcMessage("Solver type not implemented."));
            break;
          }
      } // End switch (parameters->solver)

    return (return_value);
  }

#include "NSolver.inst"
} // namespace NSFEMSolver
