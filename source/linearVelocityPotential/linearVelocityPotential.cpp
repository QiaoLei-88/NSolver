//  Created by 乔磊 on 2015/8/7.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/linearVelocityPotential/linearVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  LinearVelocityPotential<dim>::LinearVelocityPotential (
    const SmartPointer<parallel::distributed::Triangulation<dim> const > triangulation_in,
    const SmartPointer<NSFEMSolver::Parameters::AllParameters<dim> > parameters_in,
    const SmartPointer<LA::MPI::Vector> output_initial_field_ptr,
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
    velocity_infty (0.5, 0.0, 0.0)
  {}


  template <int dim>
  LinearVelocityPotential<dim>::~LinearVelocityPotential()
  {
    dof_handler.clear();
  }


  template <int dim>
  LinearVelocityPotential<dim>::Postprocessor::
  Postprocessor (const Point<3> velocity_infty_in)
    :
    velocity_infty (velocity_infty_in)
  {}

#include "linearVelocityPotential.inst.in"
}
