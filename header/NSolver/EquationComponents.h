//
//  EquationComponents.h
//  NSolver
//
//  Created by 乔磊 on 15/5/14.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__EquationComponents__
#define __NSolver__EquationComponents__

#include <deal.II/numerics/data_component_interpretation.h>

#include <string>
#include <vector>

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  class EquationComponents
  {
  public:
    // @sect4{Component description}

    // First a few variables that describe the various components of our
    // solution vector in a generic way. They have to be initialized in
    // header files in order to be used as constant expressions.
    static const unsigned int n_components             = dim + 2;
    static const unsigned int first_momentum_component = 0;
    static const unsigned int first_velocity_component = 0;
    static const unsigned int density_component        = dim;
    static const unsigned int energy_component         = dim + 1;
    static const unsigned int pressure_component       = dim + 1;

    // When generating graphical output way down in this program, we need to
    // specify the names of the solution variables as well as how the various
    // components group into vector and scalar fields. We could describe this
    // there, but in order to keep things that have to do with the Euler
    // equation localized here and the rest of the program as generic as
    // possible, we provide this sort of information in the following two
    // functions:
    static std::vector<std::string>
    component_names();

    static std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation();
  };
} // namespace NSFEMSolver

#endif /* defined(__NSolver__EquationComponents__) */
