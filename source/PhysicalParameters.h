//
//  Created by 乔磊 on 15/5/10.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#ifndef __NSolver__PhysicalParameters__
#define __NSolver__PhysicalParameters__

#include <deal.II/base/parameter_handler.h>


namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    struct PhysicalParameters
    {
      double Mach;
      double Reynolds;

      // Parameters for Sutherland's law, in Kelvin.
      double reference_temperature;
      double Sutherland_constant;

      // Angle of attack, in degree
      double angle_of_attack;
      // Angle of slip, in degree
      double angle_of_side_slip;


      double reference_chord;
      double reference_span;
      double reference_area;

      double gas_gamma;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };
  }
}
#endif
