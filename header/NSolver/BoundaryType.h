//
//  BoundaryType.h
//  NSolver
//
//  Created by 乔磊 on 15/5/13.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef NSolver_BoundaryType_h
#define NSolver_BoundaryType_h

namespace NSFEMSolver
{
  namespace Boundary
  {
    enum Type
    {
      Symmetry,
      FarField,
      NonSlipWall,
      AllPrimitiveValues,
      PressureOutlet,
      MomentumInlet,
      MMS_BC
    };
  }
}
#endif
