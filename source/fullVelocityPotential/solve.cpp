//  Created by 乔磊 on 2015/9/8.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void FullVelocityPotential<dim>::solve (double &final_residual)
  {
    TimerOutput::Scope t (computing_timer, "solve");

    SolverControl solver_control (300, 1e-10);

    LA::SolverGMRES solver (solver_control, mpi_communicator);

    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    data.elliptic = false;
#endif
    preconditioner.initialize (system_matrix, data);

    solver.solve (system_matrix, newton_update, system_rhs,
                  preconditioner);

    pcout << "  Linear system solved in " << solver_control.last_step()
          << " iterations." << std::endl;
    final_residual = solver_control.last_value();
    pcout << "  Linear residual = " << solver_control.last_value()
          << std::endl;
  }

#include "fullVelocityPotential.inst"
}
