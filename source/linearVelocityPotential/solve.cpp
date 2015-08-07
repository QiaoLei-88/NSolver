//  Created by 乔磊 on 2015/8/7.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/linearVelocityPotential/linearVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void LinearVelocityPotential<dim>::solve()
  {
    TimerOutput::Scope t (computing_timer, "solve");

    LA::MPI::Vector
    locally_owned_solution (locally_owned_dofs, mpi_communicator);

    SolverControl solver_control (dof_handler.n_dofs(), 1e-12);

    LA::SolverCG solver (solver_control, mpi_communicator);
    LA::MPI::PreconditionAMG preconditioner;

    LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    /* Trilinos defaults are good */
#endif
    preconditioner.initialize (system_matrix, data);

    solver.solve (system_matrix, locally_owned_solution, system_rhs,
                  preconditioner);

    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl;

    constraints.distribute (locally_owned_solution);

    locally_relevant_solution = locally_owned_solution;
  }

#include "linearVelocityPotential.inst.in"
}
