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

    // Find out global minimum cell size
    double local_h_min = std::numeric_limits<double>::max();
    if (parameters->use_local_time_step_size ||
        parameters->use_local_laplacian_coefficient)
      {
        typename DoFHandler<dim>::active_cell_iterator cell =
          dof_handler.begin_active();
        const typename DoFHandler<dim>::active_cell_iterator endc =
          dof_handler.end();
        for (; cell!=endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                local_h_min = std::min (cell->diameter(), local_h_min);
              }
          }
      }
    const double global_h_min = Utilities::MPI::min (local_h_min, mpi_communicator);

    unsigned int n_laplacian = 0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          ++n_laplacian;

          fe_v.reinit (cell);
          cell->get_dof_indices (dof_indices);

          // Set and adjust coefficients of pseudo and laplacian operators.
          double local_laplacian_coeff = continuation_coeff_laplacian;
          double local_time_coeff = continuation_coeff_time;
          // No adjust need on first step. Use laplacian continuation globally.
          if ((n_time_step > 0) && parameters->enable_partial_laplacian)
            {
              if (laplacian_indicator[cell->active_cell_index()] < laplacian_threshold)
                {
                  // In smooth region, use pseudo time continuation only.
                  local_laplacian_coeff = 0.0;
                  local_time_coeff = continuation_coefficient;
                  --n_laplacian;
                }
            }

          if (parameters->use_local_laplacian_coefficient && (local_laplacian_coeff > 0.0) && (n_time_step > 0))
            {
              local_laplacian_coeff *= (global_h_min/cell->diameter());
            }

          if (parameters->use_local_time_step_size && (local_time_coeff > 0.0))
            {
              local_time_coeff *= (global_h_min/cell->diameter());
            }

          if (parameters->continuation_type == Parameters::StabilizationParameters<dim>::CT_timeCFL)
            {
              local_laplacian_coeff = 0.0;
              local_time_coeff = parameters->use_local_time_step_size ?
                                 1.0/local_time_step_size[cell->active_cell_index()]
                                 :
                                 1.0/global_time_step_size;
            }
          if (local_laplacian_coeff > 0.0)
            {
              std::vector<std::vector<Tensor<1,dim> > > grad_W (n_q_points,
                                                                std::vector<Tensor<1,dim> > (EquationComponents<dim>::n_components));
              // Use laplacian term
              fe_v.get_function_gradients (current_solution, grad_W);
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  double residual = 0.0;
                  const unsigned int c = fe_v.get_fe().system_to_component_index (i).first;
                  for (unsigned int q=0; q<n_q_points; ++q)
                    {
                      residual += local_laplacian_coeff *
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
                              matrix_row[j]  += local_laplacian_coeff *
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
          if (local_time_coeff > 0.0)
            {
              Table<2,DFADD > W (n_q_points, EquationComponents<dim>::n_components);
              Table<2,double> W_old (n_q_points, EquationComponents<dim>::n_components);
              std::vector<DFADD> independent_local_dof_values (dofs_per_cell);
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  independent_local_dof_values[i] = current_solution[dof_indices[i]];
                  independent_local_dof_values[i].diff (i, dofs_per_cell);
                }
              // Use mass matrix, i.e., pseudo time continuation
              std_cxx11::array<DFADD, EquationComponents<dim>::n_components> w_conservative;
              std_cxx11::array<double, EquationComponents<dim>::n_components> w_conservative_old;

              {
                // Get function values
                for (unsigned int q=0; q<n_q_points; ++q)
                  for (unsigned int c=0; c<EquationComponents<dim>::n_components; ++c)
                    {
                      W[q][c] = 0.0;
                      W_old[q][c] = 0.0;
                    }

                for (unsigned int q=0; q<n_q_points; ++q)
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                      const unsigned int c = fe_v.get_fe().system_to_component_index (i).first;

                      W[q][c] += independent_local_dof_values[i] *
                                 fe_v.shape_value_component (i, q, c);
                      W_old[q][c] += old_solution (dof_indices[i]) *
                                     fe_v.shape_value_component (i, q, c);
                    }
              }

              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  DFADD residual = 0.0;
                  const unsigned int c = fe_v.get_fe().system_to_component_index (i).first;

                  for (unsigned int q=0; q<n_q_points; ++q)
                    {
                      EulerEquations<dim>::compute_conservative_vector (W[q], w_conservative);
                      EulerEquations<dim>::compute_conservative_vector (W_old[q], w_conservative_old);
                      residual += local_time_coeff *
                                  (w_conservative[c] - w_conservative_old[c]) *
                                  fe_v.shape_value_component (i, q, c) *
                                  fe_v.JxW (q);
                    }

                  system_matrix.add (dof_indices[i], dof_indices.size(),
                                     & (dof_indices[0]), residual.dx());
                  if (!parameters->is_steady || parameters->count_solution_diff_in_residual)
                    {
                      right_hand_side (dof_indices[i]) -= residual.val();
                    }
                  if (!parameters->is_steady)
                    {
                      physical_residual (dof_indices[i]) -= residual.val();
                    }
                }
            } // End of if (local_time_coeff > 0.0)
        }
    pcout << "n_laplacian = "
          << Utilities::MPI::sum (n_laplacian, mpi_communicator)
          << std::endl;

    return;
  }

#include "NSolver.inst"
}
