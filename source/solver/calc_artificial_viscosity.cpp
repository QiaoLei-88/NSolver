//
//  NSolver::calc_artificial_viscosity.cpp
//
//  Created by Lei Qiao on 15/9/2.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;
  template <int dim>
  void NSolver<dim>::calc_artificial_viscosity()
  {
    // Entropy viscosity
    FEValues<dim> fe_values (fe,
                             quadrature,
                             update_values |
                             update_gradients |
                             update_quadrature_points);

    const unsigned int n_q_points = quadrature.size();
    const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    std::vector<Vector<double> > W (n_q_points, Vector<double> (EquationComponents<dim>::n_components));
    std::vector<std::vector<Tensor<1,dim> > > grad_W (n_q_points,
                                                      std::vector<Tensor<1,dim> > (EquationComponents<dim>::n_components));
    std::vector<Vector<double> > W_old (n_q_points, Vector<double> (EquationComponents<dim>::n_components));

    std::vector<types::global_dof_index> global_indices_of_local_dofs (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell =
      dof_handler.begin_active();
    const typename DoFHandler<dim>::active_cell_iterator endc =
      dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            const unsigned int cell_index = cell->active_cell_index();

            fe_values.reinit (cell);
            fe_values.get_function_values (current_solution, W);
            fe_values.get_function_gradients (current_solution, grad_W);
            fe_values.get_function_values (old_solution, W_old);

            cell->get_dof_indices (global_indices_of_local_dofs);

            double rho_max (-1.0), D_h_max (-1.0), characteristic_speed_max (-1.0);

            for (unsigned int q=0; q<n_q_points; ++q)
              {
                // Here, we need to evaluate the derivatives of entropy flux respect to Euler equation independent variables $w$
                // rather than the unknown vector $W$. So we have to set up a new Sacado::Fad::DFad system.
                std_cxx11::array<Sacado::Fad::DFad<double>, EquationComponents<dim>::n_components> w_for_entropy_flux;
                for (unsigned int c=0; c<EquationComponents<dim>::n_components; ++c)
                  {
                    w_for_entropy_flux[c] = W[q][c];
                    w_for_entropy_flux[c].diff (c, EquationComponents<dim>::n_components);
                  }

                const Sacado::Fad::DFad<double> entropy = EulerEquations<dim>::template compute_entropy (w_for_entropy_flux);
                const double entroy_old = EulerEquations<dim>::template compute_entropy (W_old[q]);

                double D_h1 (0.0),D_h2 (0.0);
                D_h1 = (entropy.val() - entroy_old)/time_step;
                D_h2 = (W[q][EquationComponents<dim>::density_component] - W_old[q][EquationComponents<dim>::density_component])/
                time_step;

                //sum up divergence
                for (unsigned int d=0; d<dim; d++)
                  {
                    const Sacado::Fad::DFad<double> entropy_flux = entropy *
                                                                   w_for_entropy_flux[EquationComponents<dim>::first_velocity_component + d];
                    for (unsigned int c=0; c<EquationComponents<dim>::n_components; ++c)
                      {
                        D_h1 += entropy_flux.fastAccessDx (c) * grad_W[q][c][d];
                      }
                    D_h2 += grad_W[q][EquationComponents<dim>::first_velocity_component + d][d]
                            * W[q][EquationComponents<dim>::density_component]
                            + W[q][EquationComponents<dim>::first_velocity_component + d]
                            * grad_W[q][EquationComponents<dim>::density_component][d];
                  }
                D_h2 *= entropy.val()/W[q][EquationComponents<dim>::density_component];
                D_h_max = std::max (D_h_max, std::abs (D_h1));
                D_h_max = std::max (D_h_max, std::abs (D_h2));

                rho_max = std::max (rho_max, W[q][EquationComponents<dim>::density_component]);

                const double sound_speed
                  = EulerEquations<dim>::template compute_sound_speed (W[q]);
                const double velocity
                = EulerEquations<dim>::template compute_velocity_magnitude (W[q]);
                characteristic_speed_max = std::max (characteristic_speed_max, velocity + sound_speed);
              }
            const double entropy_visc
            = parameters->entropy_visc_cE * rho_max
            * std::pow (fe_values.get_cell()->diameter(), 1.5) * D_h_max;
            const double miu_max
            = parameters->entropy_visc_cLinear
            * fe_values.get_cell()->diameter()
            * rho_max * characteristic_speed_max;

            entropy_viscosity[cell_index] = std::min (miu_max, entropy_visc);

            // Cell size based artificial viscosity
            cellSize_viscosity[cell_index] = 1.0*std::pow (fe_values.get_cell()->diameter(), parameters->diffusion_power);

          } // End if cell is locally owned
      } // End for active cells
  } // End function

#include "NSolver.inst"
}
