//  Created by 乔磊 on 2015/8/7.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/linearVelocityPotential/linearVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void LinearVelocityPotential<dim>::compute()
  {
    pcout << "  Solving linearized velocity potential euqation for initial value" << std::endl;
    setup_system();

    pcout << "   Number of active cells:       "
          << triangulation->n_global_active_cells()
          << std::endl
          << "   Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << std::endl;

    assemble_system();
    solve();
    computing_timer.print_summary();
    pcout << std::endl;
  }

#include "linearVelocityPotential.inst.in"
}
