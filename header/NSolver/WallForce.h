//
//  WallForce.h
//  NSolver
//
//  Created by 乔磊 on 15/5/10.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#ifndef __NSolver__WallForce__
#define __NSolver__WallForce__

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

namespace NSFEMSolver
{
  using namespace dealii;

  struct WallForce
  {
    void
    clear();
    void
    mpi_sum(MPI_Comm const &mpi_communicator);
    WallForce
    operator+(WallForce const &op_r);

    double   lift;
    double   drag;
    Point<3> force;
    Point<3> moment;
  };
} // namespace NSFEMSolver
#endif
