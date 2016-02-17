//
//  MMS.h
//  NSolver
//
//  Created by 乔磊 on 15/8/16.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__types__
#define __NSolver__types__

#include <deal.II/lac/generic_linear_algebra.h>


//#define USE_TRILINOS_LA
#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
} // namespace LA

namespace NSFEMSolver
{
  using namespace dealii;

  typedef LA::MPI::Vector       NSVector;
  typedef LA::MPI::SparseMatrix NSMatrix;
} // namespace NSFEMSolver

#endif
