//
//  NSolver::apply_laplacian_continuation.cpp
//
//  Created by Lei Qiao on 15/11/5.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  void
  NSolver<dim>::apply_laplacian_continuation()
  {
    typedef Sacado::Fad::DFad<double> DFADD;

    // Apply the laplacian operator
    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();
    std::vector<types::global_dof_index> dof_indices (dofs_per_cell);

    const UpdateFlags update_flags = update_values
                                     | update_gradients
                                     | update_quadrature_points
                                     | update_JxW_values;
    FEValues<dim> fe_v (*mapping_ptr, fe, quadrature, update_flags);

    std::vector<std::vector<Tensor<1,dim> > > grad_W (n_q_points,
                                                      std::vector<Tensor<1,dim> > (EquationComponents<dim>::n_components));
    unsigned int n_laplacian = 0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          const bool use_laplacian =
            n_time_step == 0 ||
            laplacian_indicator[cell->active_cell_index()] >= laplacian_threshold;
          n_laplacian += use_laplacian;
          const double factor = use_laplacian
                                ? 1.0
                                : 0.01;

          fe_v.reinit (cell);
          cell->get_dof_indices (dof_indices);
          fe_v.get_function_gradients (current_solution, grad_W);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              double residual = 0.0;
              const unsigned int c = fe_v.get_fe().system_to_component_index (i).first;
              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  residual += laplacian_coefficient * factor *
                              (grad_W[q][c] *
                               fe_v.shape_grad_component (i, q, c)) *
                              fe_v.JxW (q);
                }
              std::vector<double> matrix_row (dofs_per_cell, 0.0);
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  if (c == fe_v.get_fe().system_to_component_index (j).first)
                    {
                      for (unsigned int q=0; q<n_q_points; ++q)
                        {
                          matrix_row[j]  += laplacian_coefficient * factor *
                                            (fe_v.shape_grad_component (j, q, c)
                                             * fe_v.shape_grad_component (i, q, c)) *
                                            fe_v.JxW (q);
                        }
                    }
                }
              system_matrix.add (dof_indices[i], dof_indices.size(),
                                 & (dof_indices[0]), & (matrix_row[0]));
              right_hand_side (dof_indices[i]) -= residual;
            }
        }
    pcout << "n_laplacian = "
          << Utilities::MPI::sum (n_laplacian, mpi_communicator)
          << std::endl;

    return;
  }

#include "NSolver.inst"
}
