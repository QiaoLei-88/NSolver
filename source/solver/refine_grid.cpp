//
//  NSolver::refine_grid.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II turorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


  // @sect4{NSolver::refine_grid}

  // Here, we use the refinement indicators computed before and refine the
  // mesh. At the beginning, we loop over all cells and mark those that we
  // think should be refined:
  template <int dim>
  void
  NSolver<dim>::refine_grid()
  {
    switch (parameters->refinement_indicator)
      {
      case Parameters::Refinement<dim>::Gradient:
      {
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
          if (cell->is_locally_owned())
            {
              cell->clear_coarsen_flag();
              cell->clear_refine_flag();

              if ((cell->level() < parameters->shock_levels) &&
                  (std::fabs (refinement_indicators (cell_no)) > parameters->shock_val))
                {
                  cell->set_refine_flag();
                }
              else if ((cell->level() > 0) &&
                       (std::fabs (refinement_indicators (cell_no)) < 0.75*parameters->shock_val))
                {
                  cell->set_coarsen_flag();
                }
            }
        break;
      }
      case Parameters::Refinement<dim>::Kelly:
      {
        parallel::distributed::GridRefinement::
        refine_and_coarsen_fixed_number (triangulation,
                                         refinement_indicators,
                                         parameters->refine_fraction,
                                         parameters->coarsen_fraction,
                                         parameters->max_cells);
        break;
      }
      default:
      {
        Assert (false, ExcNotImplemented());
        break;
      }
      }

    // Then we need to transfer the various solution vectors from the old to
    // the new grid while we do the refinement. The SolutionTransfer class is
    // our friend here; it has a fairly extensive documentation, including
    // examples, so we won't comment much on the following code. The last
    // three lines simply re-set the sizes of some other vectors to the now
    // correct size:

    NSVector tmp_vector;
    tmp_vector.reinit (old_solution, true);
    tmp_vector = predictor;
    // transfer_in needs vectors with ghost cells.
    std::vector<const NSVector * > transfer_in;
    transfer_in.push_back (&old_solution);
    transfer_in.push_back (&tmp_vector);

    parallel::distributed::SolutionTransfer<dim, NSVector> soltrans (dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    soltrans.prepare_for_coarsening_and_refinement (transfer_in);

    triangulation.execute_coarsening_and_refinement();

    setup_system();

    // Transfer data out
    {
      std::vector<NSVector * > transfer_out;
      NSVector interpolated_old_solution (predictor);
      NSVector interpolated_predictor (predictor);
      // transfer_out needs vectors without ghost cells.
      transfer_out.push_back (&interpolated_old_solution);
      transfer_out.push_back (&interpolated_predictor);
      soltrans.interpolate (transfer_out);
      old_solution = interpolated_old_solution;
      predictor = interpolated_predictor;
      current_solution = old_solution;
    }
  }

#include "NSolver.inst"
}
