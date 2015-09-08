//  Created by 乔磊 on 2015/9/8.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  FullVelocityPotential<dim>::FullVelocityPotential (
    const SmartPointer<parallel::distributed::Triangulation<dim> const > triangulation_in,
    const SmartPointer<NSFEMSolver::Parameters::AllParameters<dim> > parameters_in,
    MPI_Comm mpi_communicator_in)
    :
    mpi_communicator (mpi_communicator_in),
    parameters (parameters_in),
    triangulation (triangulation_in),
    dof_handler (*triangulation_in),
    fe (parameters_in->init_fe_degree),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process (mpi_communicator)
            == 0)),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::never,
                     TimerOutput::wall_times),
    Mach_infty (parameters_in->Mach),
    Mach_infty_square (Mach_infty *Mach_infty),
    gas_gamma (parameters_in->gas_gamma),
    gm1 (gas_gamma-1.0),
    const_a (1.0+0.5*gm1 *Mach_infty *Mach_infty)
  {
    // Compute free stream condition.
    // Axi definition:
    //   X in flow direction
    //   Y up
    //   Z size
    // Linear velocity potential equation make nonsense in high Mach number
    const double Mach = std::min (parameters->Mach, 0.8);

    if (dim == 3)
      {
        velocity_infty[2] = Mach * std::sin (parameters->angle_of_side_slip);
      }
    const double velocity_in_symm_plan =
      Mach * std::cos (parameters->angle_of_side_slip);
    velocity_infty[1] = velocity_in_symm_plan *
                        std::sin (parameters->angle_of_attack);
    velocity_infty[0] = velocity_in_symm_plan *
                        std::cos (parameters->angle_of_attack);
  }


  template <int dim>
  FullVelocityPotential<dim>::~FullVelocityPotential()
  {
    dof_handler.clear();
  }

#include "fullVelocityPotential.inst"
}
