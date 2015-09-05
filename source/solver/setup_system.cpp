//
//  NSolver::setup_system.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;

  // @sect4{NSolver::setup_system}
  //
  // The following (easy) function is called each time the mesh is
  // changed. All it does is to resize the Trilinos matrix according to a
  // sparsity pattern that we generate as in all the previous tutorial
  // programs.
  template <int dim>
  void NSolver<dim>::setup_system()
  {
    dof_handler.clear();
    dof_handler.distribute_dofs (fe);

    dof_handler_visc.clear();
    dof_handler_visc.distribute_dofs (fe_visc);

    if (parameters->output_sparsity_pattern)
      {
        ++n_sparsity_pattern_out;
        TrilinosWrappers::SparsityPattern sparsity_pattern (locally_owned_dofs,
                                                            mpi_communicator);
        DoFTools::make_sparsity_pattern (dof_handler,
                                         sparsity_pattern,
                                         /*const ConstraintMatrix constraints = */ ConstraintMatrix(),
                                         /*const bool keep_constrained_dofs = */ true,
                                         Utilities::MPI::this_mpi_process (mpi_communicator));
        sparsity_pattern.compress();

        const std::string file_name = "sparsity_pattern."
                                      + Utilities::int_to_string (n_sparsity_pattern_out,4)
                                      + ".origin";
        std::ofstream out (file_name.c_str());
        sparsity_pattern.print_gnuplot (out);
      }

    switch (parameters->renumber_dofs)
      {
      case Parameters::AllParameters<dim>::None :
      {
        break;
      }
      case Parameters::AllParameters<dim>::RCM :
      {
        DoFRenumbering::Cuthill_McKee (dof_handler,
                                       /* reversed_numbering = */ true);
        break;
      }
      case Parameters::AllParameters<dim>::RCM_WithStartPoint :
      {
        std::vector<types::global_dof_index> dof_indices (fe.dofs_per_cell);
        Point<dim> target_point;
        for (unsigned int id=0; id<dim; ++id)
          {
            target_point[id] = parameters->renumber_start_point[id];
          }

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        unsigned int cell_index (0);
        for (; cell!=endc; ++cell, ++cell_index)
          if (cell->is_locally_owned())
            {
              if (target_point.distance (cell->center()) < 0.01)
                {
                  cell->get_dof_indices (dof_indices);
                  break;
                }
            }

        DoFRenumbering::Cuthill_McKee (dof_handler,
                                       /* reversed_numbering = */ true,
                                       /* use_constraints = */    false ,
                                       /* starting_indices */     dof_indices);
        break;
      }
      default:
      {
        Assert (false, ExcNotImplemented());
        break;
      }
      }

    locally_owned_dofs.clear();
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    locally_owned_visc_index.clear();
    locally_owned_visc_index = dof_handler_visc.locally_owned_dofs();

    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    locally_relevant_visc_index.clear();
    DoFTools::extract_locally_relevant_dofs (dof_handler_visc,
                                             locally_relevant_visc_index);

    TrilinosWrappers::SparsityPattern sparsity_pattern (locally_owned_dofs,
                                                        mpi_communicator);
    DoFTools::make_sparsity_pattern (dof_handler,
                                     sparsity_pattern,
                                     /*const ConstraintMatrix constraints = */ ConstraintMatrix(),
                                     /*const bool keep_constrained_dofs = */ true,
                                     Utilities::MPI::this_mpi_process (mpi_communicator));
    sparsity_pattern.compress();
    if (parameters->output_sparsity_pattern &&
        parameters->renumber_dofs != Parameters::AllParameters<dim>::None)
      {
        const std::string file_name = "sparsity_pattern."
                                      + Utilities::int_to_string (n_sparsity_pattern_out,4)
                                      + ".renumbered";
        std::ofstream out (file_name.c_str());
        sparsity_pattern.print_gnuplot (out);
      }
    system_matrix.reinit (sparsity_pattern);

    // Initialize vectors
    locally_owned_solution.reinit (locally_owned_dofs, mpi_communicator);
    current_solution.reinit (locally_relevant_dofs, mpi_communicator);

    // const bool fast = true means leave its content untouched.
    old_solution.reinit (current_solution, /*const bool fast = */ true);
    old_old_solution.reinit (current_solution, true);
    predictor.reinit (locally_owned_solution, true);

    right_hand_side.reinit (locally_owned_solution, true);
    newton_update.reinit (locally_owned_solution, true);

    residual_for_output.reinit (current_solution, true);

    entropy_viscosity.reinit (triangulation.n_active_cells());
    cellSize_viscosity.reinit (triangulation.n_active_cells());
    refinement_indicators.reinit (triangulation.n_active_cells());

    viscosity_coeff.reinit (locally_relevant_visc_index, mpi_communicator);
  }

#include "NSolver.inst"
}
