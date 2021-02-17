//  Created by 乔磊 on 2015/9/8.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void
  FullVelocityPotential<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    const QGauss<dim>     quadrature_formula(3);
    const QGauss<dim - 1> face_quadrature_formula(2);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<Tensor<1, dim>>          solution_gradients(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs    = 0;

          fe_values.reinit(cell);
          fe_values.get_function_gradients(locally_relevant_solution,
                                           solution_gradients);
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const double density =
                internal::compute_density<dim>(solution_gradients[q_point],
                                               gm1,
                                               Mach_infty_square);
              const double density_1mc = std::pow(density, 1.0 - gm1);
              const double const_b     = solution_gradients[q_point] *
                                     solution_gradients[q_point] * density_1mc;
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        ((fe_values.shape_grad(i, q_point) * density *
                            fe_values.shape_grad(j, q_point) -
                          fe_values.shape_grad(i, q_point) * const_b *
                            fe_values.shape_grad(j, q_point)) *
                         fe_values.JxW(q_point));
                    }
                  cell_rhs(i) -=
                    (fe_values.shape_grad(i, q_point) * density *
                     solution_gradients[q_point] * fe_values.JxW(q_point));
                }
            }
          // Aplly Neumann boundary condition
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              {
                switch (
                  parameters->boundary_conditions[cell->face(f)->boundary_id()]
                    .kind)
                  {
                    case NSFEMSolver::Boundary::NonSlipWall:
                    case NSFEMSolver::Boundary::Symmetry: // i.e. SlipWall
                      {
                        // Zero Neumann boundary condition. Do nothing.
                        break;
                      }
                    case NSFEMSolver::Boundary::FarField:
                      {
                        fe_face_values.reinit(cell, f);
                        for (unsigned int q_point = 0;
                             q_point < n_face_q_points;
                             ++q_point)
                          {
                            const double face_flux =
                              (fe_face_values.normal_vector(q_point) *
                               velocity_infty) *
                              fe_face_values.JxW(q_point);
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                cell_rhs(i) +=
                                  face_flux *
                                  fe_face_values.shape_value(i, q_point);
                              }
                          }
                        break;
                      }
                    default:
                      {
                        break;
                      }
                  }
              }
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

#include "fullVelocityPotential.inst"
} // namespace velocityPotential
