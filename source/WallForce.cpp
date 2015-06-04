//
//  WallForce.cpp
//  NSolver
//
//  Created by 乔磊 on 15/5/10.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#include <NSolver/WallForce.h>

namespace NSFEMSolver
{
  using namespace dealii;

  void WallForce::clear()
  {
    lift = 0.0;
    drag = 0.0;
    for (unsigned int d=0; d<3; ++d)
      {
        force[d] = 0.0;
        moment[d] = 0.0;
      }
  }

  WallForce WallForce::operator+ (WallForce const &op_r)
  {
    WallForce rv;
    rv.lift =this->lift + op_r.lift;
    rv.drag =this->drag + op_r.drag;
    rv.force =this->force + op_r.force;
    rv.moment =this->moment + op_r.moment;
    return (rv);
  }

  void WallForce::mpi_sum (MPI_Comm const &mpi_communicator)
  {
    Utilities::MPI::sum (lift, mpi_communicator);
    Utilities::MPI::sum (drag, mpi_communicator);
    for (unsigned int d=0; d<3; ++d)
      {
        Utilities::MPI::sum (force[d], mpi_communicator);
        Utilities::MPI::sum (moment[d], mpi_communicator);
      }
  }

}
