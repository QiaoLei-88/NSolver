//  Created by 乔磊 on 2015/9/12.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void FullVelocityPotential<dim>::apply_kutta_condition (types::global_dof_index &TE_dof_index)
  {
    // only works in 2D
    if (dim != 2)
      {
        return;
      }

    const Point<dim> trailind_edge (1.0, 0.0);

    const Quadrature<dim>    quadrature_formula (fe.get_unit_support_points());
    const QTrapez<dim-1>  face_quadrature_formula;

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_normal_vectors);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    // Find out boundary cells share the trailing edge
    // Try boundary cells only first. Hopefully, the equation will correct
    // values on other cell that share the trailing edge automatically
    Tensor<1, dim> average_velocity;
    std::vector<Tensor<1, dim> > solution_gradients (n_q_points);
    std::vector<double> solution_values (n_q_points);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<1, dim> > velocity_on_TE;
    std::vector<Tensor<1, dim> > norms_on_TE;
    // DoF indices and positions of neighboring vertices
    std::map<types::global_dof_index, Tensor<1, dim> > neighboring_vertex_info;


    neighboring_vertex_info.clear();
    velocity_on_TE.clear();
    norms_on_TE.clear();
    double value_on_TE (-1984);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned() && cell->at_boundary())
        {
          if (trailind_edge.distance (cell->center()) < 0.012)
            {
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
                      velocity_on_TE.push_back (solution_gradients[q_point]);
                      value_on_TE = solution_values[q_point];
                      TE_dof_index = local_dof_indices[q_point];
                      std::cerr << "TE vertex local index = " << q_point << std::endl;
                      // Collect information for neighboring DoFs
                      const unsigned int previous = (q_point + (((q_point&0x01)<<1)|0x01))%n_q_points;
                      neighboring_vertex_info[local_dof_indices[previous]]
                        = fe_values.quadrature_point (previous);
                      const unsigned int next = (q_point + 2)%n_q_points;
                      neighboring_vertex_info[local_dof_indices[next]]
                        = fe_values.quadrature_point (next);
                      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                        if (cell->face (f)->at_boundary())
                          {
                            fe_face_values.reinit (cell, f);
                            norms_on_TE.push_back (fe_face_values.normal_vector (0));
                            break;
                          }
                    }
                }
            }
        }
    // Evaluate average velocity on trailing edge among trailing edge cells on
    // boundary
    std::cerr << norms_on_TE.size() << ", "
              << velocity_on_TE.size()
              << std::endl;
    Assert (norms_on_TE.size() == velocity_on_TE.size(),
            ExcMessage ("Number of norms and velocity mismatch"));

    Tensor<1, dim> average_direction;
    average_direction = 0.0;
    for (unsigned int i=0; i<norms_on_TE.size(); ++i)
      {
        average_direction -= norms_on_TE[i];
      }
    average_direction /= average_direction.norm();
    std::cerr << "average_direction = "
              << average_direction[0] << ","
              << average_direction[1] << std::endl;
    // Average tangential velocity
    double velocity_porjection = 0.0;
    for (unsigned int i=0; i<velocity_on_TE.size(); ++i)
      {
        velocity_porjection += average_direction * velocity_on_TE[i];
      }
    velocity_porjection /= static_cast<double> (velocity_on_TE.size());
    average_velocity = velocity_porjection * average_direction;
    std::cerr << "average_velocity = "
              << average_velocity << std::endl;
    pcout << "Number of velocity on TE:" << norms_on_TE.size() << std::endl;

    std::cerr << "Number of corrected DoFs = " << neighboring_vertex_info.size() << std::endl;
    // Compute DoF values on trailing edge vertex's neighboring vertices
    std::cerr << std::endl
              << "DoF position:" << std::endl;

    for (typename std::map<types::global_dof_index, Tensor<1,dim> >::iterator it
         = neighboring_vertex_info.begin();
         it != neighboring_vertex_info.end();
         ++it)
      {
        std::cerr << it->second << std::endl;
        locally_owned_solution[it->first]
          = value_on_TE +
            ((it->second - trailind_edge) * average_velocity);
      }
    // Update and sync new solution
    locally_owned_solution.compress (VectorOperation::insert);
  }

#include "fullVelocityPotential.inst"
}
