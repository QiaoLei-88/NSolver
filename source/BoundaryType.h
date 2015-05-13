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
      inflow_boundary,
      outflow_boundary,
      no_penetration_boundary,
      pressure_boundary,
      Riemann_boundary,
      MMS_BC,
      SlipWall,
      Symmetry = SlipWall,
      FarField,
      PressureOutlet,
      VelocityInlet,
    };
  }
}
#endif
