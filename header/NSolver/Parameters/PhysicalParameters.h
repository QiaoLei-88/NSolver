//
//  Created by 乔磊 on 15/5/10.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#ifndef __NSolver__PhysicalParameters__
#define __NSolver__PhysicalParameters__

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <cmath>


namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    struct PhysicalParameters
    {
      unsigned int space_dimension;
      double Mach;
      double Reynolds;

      enum EqautionType
      {
        Euler,
        NavierStokes
      };
      EqautionType equation_type;

      // Parameters for Sutherland's law, in Kelvin.
      double reference_temperature;
      double Sutherland_constant;

      // Angle of attack, read in degree but store in rad.
      double angle_of_attack;
      // Angle of slip, read in degree but store in rad.
      double angle_of_side_slip;

      Point<3> moment_center;

      double reference_chord;
      double reference_span;
      double reference_area;

      double gas_gamma;

      Point<3> gravity;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };
  }
}
#endif
