//
//  EquationComponents.cpp
//  NSolver
//
//  Created by 乔磊 on 15/5/14.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/EquationComponents.h>

namespace NSFEMSolver
{
  using namespace dealii;

  // Declare the static members so they can be referenced in other source files.
  template <int dim>
  const unsigned int EquationComponents<dim>::n_components;
  template <int dim>
  const unsigned int EquationComponents<dim>::first_momentum_component;
  template <int dim>
  const unsigned int EquationComponents<dim>::first_velocity_component;
  template <int dim>
  const unsigned int EquationComponents<dim>::density_component;
  template <int dim>
  const unsigned int EquationComponents<dim>::energy_component;
  template <int dim>
  const unsigned int EquationComponents<dim>::pressure_component;

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
