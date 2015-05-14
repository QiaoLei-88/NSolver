//
//  EquationComponents.cpp
//  NSolver
//
//  Created by 乔磊 on 15/5/14.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include "EquationComponents.h"

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  std::vector<std::string>
  EquationComponents<dim>::component_names()
  {
    std::vector<std::string> names (dim, "velocity");
    names.push_back ("density");
    names.push_back ("pressure");

    return names;
  }


  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  EquationComponents<dim>::component_interpretation()
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_scalar);
    data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_scalar);

    return data_component_interpretation;
  }

  template class EquationComponents<2>;
  template class EquationComponents<3>;
}
