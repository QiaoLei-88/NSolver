//  Created by 乔磊 on 2015/9/8.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void FullVelocityPotential<dim>::setup_system()
  {
    TimerOutput::Scope t (computing_timer, "setup");

    dof_handler.distribute_dofs (fe);

    locally_owned_dofs.clear();
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_owned_solution.reinit (locally_owned_dofs, mpi_communicator);
    newton_update.reinit (locally_owned_dofs, mpi_communicator);
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);

    // VectorTools::interpolate (*mapping_ptr, dof_handler,
    //                           parameters->initial_conditions, locally_owned_solution);
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);
    locally_relevant_solution.reinit (locally_owned_dofs,
                                      locally_relevant_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    // Full velocity equation only has Neumann boundary conditions,
    // so we have to constrains one DoF to eliminate singularity.
    // Since we are only interesting in gradient but not value of velocity
    // potential, we can set the constrained DoF to arbitrary value.
    // Here we use the default value value zero.
    // Because we want to enforce Kutta condition on trailing edge,
    // we have to constrain the DoF corresponding to trailing edge vertex.
    types::global_dof_index TE_dof_index = 0;
    apply_kutta_condition (TE_dof_index);
    if (locally_owned_dofs.is_element (TE_dof_index))
      {
        constraints.add_line (TE_dof_index);
      }
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

#include "fullVelocityPotential.inst"
}
