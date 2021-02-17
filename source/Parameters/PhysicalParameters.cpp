//
//  Created by 乔磊 on 15/5/10.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#include <NSolver/Parameters/PhysicalParameters.h>

namespace NSFEMSolver
{
  namespace Parameters
  {
    using namespace dealii;

    void
    PhysicalParameters::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("physical parameters");
      {
        prm.declare_entry("space dimension",
                          "2",
                          Patterns::Integer(2, 4),
                          "Problem space dimension");

        prm.declare_entry("Mach",
                          "0.5",
                          Patterns::Double(0),
                          "Free stream Mach number");
        prm.declare_entry("Reynolds",
                          "10000",
                          Patterns::Double(0),
                          "Free stream Reynolds number");

        prm.declare_entry(
          "equation type",
          "Euler",
          Patterns::Anything(),
          "Choose governing equation from <Euler|NavierStokes>");

        prm.declare_entry(
          "reference temperature",
          "273.15",
          Patterns::Double(0),
          "Reference temperature for Sutherland's law, in Kelvin");
        prm.declare_entry("Sutherland constant",
                          "110.4",
                          Patterns::Double(0),
                          "Sutherland constant, in Kelvin");

        prm.declare_entry("angle of attack",
                          "0.0",
                          Patterns::Double(-90, +90),
                          "Angle of attack, in degree");
        prm.declare_entry("angle of side slip",
                          "0.0",
                          Patterns::Double(-90, +90),
                          "Angle of side slip, in degree");

        prm.declare_entry("moment center x",
                          "0.0",
                          Patterns::Double(),
                          "Coordinate x of moment center");
        prm.declare_entry("moment center y",
                          "0.0",
                          Patterns::Double(),
                          "Coordinate y of moment center");
        prm.declare_entry("moment center z",
                          "0.0",
                          Patterns::Double(),
                          "Coordinate z of moment center");

        prm.declare_entry("reference chord",
                          "1.0",
                          Patterns::Double(0),
                          "Reference chord length");
        prm.declare_entry("reference span",
                          "1.0",
                          Patterns::Double(0),
                          "Reference span");
        prm.declare_entry("reference area",
                          "1.0",
                          Patterns::Double(0),
                          "Reference area");

        prm.declare_entry("gas gamma",
                          "1.4",
                          Patterns::Double(1),
                          "Gas heat capacity ratio");

        prm.declare_entry("gravity x",
                          "0.0",
                          Patterns::Double(),
                          "x component of gravity");
        prm.declare_entry("gravity y",
                          "0.0",
                          Patterns::Double(),
                          "y component of gravity");
        prm.declare_entry("gravity z",
                          "0.0",
                          Patterns::Double(),
                          "z component of gravity");
      }
      prm.leave_subsection();
    }

    void
    PhysicalParameters::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("physical parameters");
      {
        double const deg_to_rad = std::atan(1.0) / 45.0;

        space_dimension = prm.get_integer("space dimension");

        Mach     = prm.get_double("Mach");
        Reynolds = prm.get_double("Reynolds");
        {
          const std::string prm_buf = prm.get("equation type");
          if (prm_buf == "Euler")
            {
              equation_type = Euler;
            }
          else if (prm_buf == "NavierStokes")
            {
              equation_type = NavierStokes;
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }
        reference_temperature = prm.get_double("reference temperature");
        Sutherland_constant   = prm.get_double("Sutherland constant");

        angle_of_attack    = deg_to_rad * prm.get_double("angle of attack");
        angle_of_side_slip = deg_to_rad * prm.get_double("angle of side slip");

        moment_center[0] = prm.get_double("moment center x");
        moment_center[1] = prm.get_double("moment center y");
        moment_center[2] = prm.get_double("moment center z");

        reference_chord = prm.get_double("reference chord");
        reference_span  = prm.get_double("reference span");
        reference_area  = prm.get_double("reference area");

        gas_gamma = prm.get_double("gas gamma");

        gravity[0] = prm.get_double("gravity x");
        gravity[1] = prm.get_double("gravity y");
        gravity[2] = prm.get_double("gravity z");
      }
      prm.leave_subsection();
    }
  } // namespace Parameters
} // namespace NSFEMSolver
