//  Created by 乔磊 on 2015/8/7.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/linearVelocityPotential.h>

namespace velocityPotential
{
  using namespace dealii;

  template <int dim>
  void LinearVelocityPotential<dim>::transfer_solution (
    const FESystem<dim>     &fe_NS,
    const DoFHandler<dim>   &dof_handler_NS,
    LA::MPI::Vector         &NS_solution) const
  {
    // Set quadrature points on NS FE supporting points.
    // Then extract desired values on these quadrature points.
    // If the quadrature points and the supporting points are arranged in
    // the same order, then we can loop through all the quadrature points and
    // assign corresponding values the NS solution vector.

    const FE_Q<dim> &fe_potential = fe;
    const DoFHandler<dim> &dof_handler_potential =dof_handler;
    const LA::MPI::Vector &potential_solution = locally_relevant_solution;

    // Here we assume all Finite Element types are the same in the NS system.
    Quadrature<dim> quadrature_on_NS_support_points (fe_NS.base_element (0).get_unit_support_points());

    FEValues<dim> fe_values_potential (fe_potential, quadrature_on_NS_support_points,
                                       update_values | update_gradients |
                                       update_quadrature_points);

    const unsigned int n_q_points = quadrature_on_NS_support_points.size();

    const unsigned int dofs_per_cell = fe_NS.dofs_per_cell;
    std::vector<types::global_dof_index> global_indices_of_local_dofs (dofs_per_cell);
    std::vector<Tensor<1,dim,double> > potential_grad_on_cell (n_q_points);

    // The two DoFHandlers are initialized from the same triangulation
    typename DoFHandler<dim>::active_cell_iterator
    cell_potential = dof_handler_potential.begin_active(),
    cell_NS = dof_handler_NS.begin_active(),
    endc_NS = dof_handler_NS.end();
    for (; cell_NS!=endc_NS; ++cell_NS, ++cell_potential)
      {
        Assert (cell_potential != dof_handler_potential.end(),
                ExcMessage ("Reached end of tria for potential equation before NS equation!"));
        if (cell_NS->is_locally_owned())
          {
            fe_values_potential.reinit (cell_potential);

            fe_values_potential.get_function_gradients (
              potential_solution,
              potential_grad_on_cell);

            cell_NS->get_dof_indices (global_indices_of_local_dofs);
            // Since the quadrature is extracted according to the target FE supporting
            // points, we just have to loop through all the quadrature points.
            // We assume that the quadrature points and supporting points are
            // arranged in the same order.
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
              {
                double Mach_local = 0.0;
                for (unsigned d=0; d<dim; ++d)
                  {
                    const double velocity =
                      velocity_infty[d] + potential_grad_on_cell[q_point][d];

                    Mach_local += velocity * velocity;
                    const unsigned int system_index = fe_NS.component_to_system_index (d, q_point);
                    NS_solution[global_indices_of_local_dofs[system_index]] = velocity;
                  }

                const double gas_gamma (1.4);
                const double Mach_infty (velocity_infty.square());
                const double Mach_ratio =
                  (1.0 + 0.5* (gas_gamma-1.0) * Mach_infty) /
                  (1.0 + 0.5* (gas_gamma-1.0) * Mach_local);
                {
                  // Density component
                  const unsigned int system_index = fe_NS.component_to_system_index (dim, q_point);
                  NS_solution[global_indices_of_local_dofs[system_index]] =
                    std::pow (Mach_ratio, 1.0/ (gas_gamma - 1.0));
                }
                {
                  //Pressure component
                  const unsigned int system_index = fe_NS.component_to_system_index (dim+1, q_point);
                  NS_solution[global_indices_of_local_dofs[system_index]] =
                    std::pow (Mach_ratio, gas_gamma/ (gas_gamma - 1.0)) / gas_gamma;
                }
              }
          }
      }
    NS_solution.compress (VectorOperation::insert);
  }

#include "linearVelocityPotential.inst"
}
