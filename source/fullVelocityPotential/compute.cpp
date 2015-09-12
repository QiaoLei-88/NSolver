//  Created by 乔磊 on 2015/9/8.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void FullVelocityPotential<dim>::compute()
  {
    pcout << "Solving full velocity potential equation for initial value" << std::endl;
    setup_system();

    pcout << "  Number of active cells:       "
          << triangulation->n_global_active_cells()
          << std::endl
          << "  Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << std::endl;

    bool newton_iter_converged = false;
    bool linear_solver_diverged (true);
    unsigned int const nonlin_iter_threshold (10);
    double const nonlinear_tolerance (-10.0);
    unsigned int nonlin_iter = 0;
    double res_norm;

    pcout << std::endl;
    output_results();
    do // Newton iteration
      {
        system_matrix = 0;
        system_rhs = 0;
        newton_update = 0;
        assemble_system();
        constraints.distribute (newton_update);
        res_norm = system_rhs.l2_norm();
        pcout << "Nonlinear step = " << nonlin_iter
              << ", res_norm = " << res_norm
              << std::endl;
        newton_iter_converged
          = (std::log10 (res_norm) < nonlinear_tolerance);
        if (newton_iter_converged)
          {
            pcout << std::endl
                  << "Nonlinear iteration converged at step = " << nonlin_iter
                  << ", with res_norm = " << res_norm
                  << std::endl;
            break;
          }
        double final_residual;
        solve (final_residual);
        constraints.distribute (newton_update);
        locally_owned_solution += newton_update;
        locally_relevant_solution = locally_owned_solution;
        output_results();
        {
          types::global_dof_index dummy;
          apply_kutta_condition (dummy);
        }
        locally_relevant_solution = locally_owned_solution;
        linear_solver_diverged = std::isnan (final_residual);
        ++nonlin_iter;
        output_results();
      }
    while (nonlin_iter < nonlin_iter_threshold
           && (!linear_solver_diverged));

    computing_timer.print_summary();
    pcout << std::endl;
  }

#include "fullVelocityPotential.inst"
}
