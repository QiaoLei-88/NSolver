//  Created by 乔磊 on 2015/8/7.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/linearVelocityPotential/linearVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void LinearVelocityPotential<dim>::setup_system()
  {
    TimerOutput::Scope t (computing_timer, "setup");

    dof_handler.distribute_dofs (fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    locally_relevant_solution.reinit (locally_owned_dofs,
                                      locally_relevant_dofs, mpi_communicator);
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
                                              ZeroFunction<dim>(),
                                              constraints);
    constraints.close();

    DynamicSparsityPattern dsp (locally_relevant_dofs);

    DoFTools::make_sparsity_pattern (dof_handler, dsp,
                                     constraints, false);
    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);
  }

#include "linearVelocityPotential.inst"
}
