//  Created by 乔磊 on 2015/9/12.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void FullVelocityPotential<dim>::apply_kutta_condition()
  {
    // only works in 2D
    if (dim != 2)
      {
        return;
      }

    const Point<dim> trailind_edge (1.0, 0.0);

    const Quadrature<dim>    quadrature_formula (fe.get_unit_support_points());

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    // Find out boundary cells share the trailing edge
    // Try boundary cells only first. Hopefully, the equation will correct
    // values on other cell that share the trailing edge automatically
    Tensor<1, dim> average_velocity;
    std::vector<Tensor<1, dim> > solution_gradients (n_q_points);
    std::vector<double> solution_values (n_q_points);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Point<dim> > position_of_neighboring_dofs;
    std::map<types::global_dof_index, Tensor<1, dim> > neighboring_vertex_info;


    neighboring_vertex_info.clear();
    average_velocity = 0.0;
    int n_velocity = 0;
    double value_on_TE (-1984);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned() && cell->at_boundary())
        {
          if (trailind_edge.distance (cell->center()) < 0.012)
            {
              // Evaluate average velocity on trailing edge among trailing edge cells on
              // boundary
              fe_values.reinit (cell);
              fe_values.get_function_gradients (locally_relevant_solution,
                                                solution_gradients);
              fe_values.get_function_values (locally_relevant_solution,
                                             solution_values);
              cell->get_dof_indices (local_dof_indices);
              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  if (trailind_edge.distance (fe_values.quadrature_point (q_point)) < 1.0e-6)
                    {
                      ++n_velocity;
                      average_velocity += solution_gradients[q_point];
                      value_on_TE = solution_values[q_point];

                      // Collect information for neighboring DoFs
                      const unsigned int previous = ((q_point+n_q_points) - 1)%n_q_points;
                      neighboring_vertex_info[local_dof_indices[previous]]
                        = fe_values.quadrature_point (previous);
                      const unsigned int next = (q_point + 1)%n_q_points;
                      neighboring_vertex_info[local_dof_indices[next]]
                        = fe_values.quadrature_point (next);
                    }
                }
            }
        }
    average_velocity /= static_cast<double> (n_velocity);
    pcout << "Number of velocity on TE:" << n_velocity << std::endl;

    // Compute DoF values on trailing edge vertex's neighboring vertices
    for (typename std::map<types::global_dof_index, Tensor<1,dim> >::iterator it
         = neighboring_vertex_info.begin();
         it != neighboring_vertex_info.end();
         ++it)
      {
        locally_owned_solution[it->first]
          = value_on_TE +
            ((it->second - trailind_edge) * average_velocity);
      }
    // Update and sync new solution
    locally_owned_solution.compress (VectorOperation::insert);
  }

#include "fullVelocityPotential.inst"
}
