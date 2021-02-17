//  Created by 乔磊 on 2015/9/8.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  FullVelocityPotential<dim>::Postprocessor::
  Postprocessor (const double Mach_infty_square_in,
                 const double gas_gamma_in)
    :
    Mach_infty_square (Mach_infty_square_in),
    gas_gamma (gas_gamma_in),
    gm1 (gas_gamma - 1.0)
  {}

  template <int dim>
  void
  FullVelocityPotential<dim>::Postprocessor::
  evaluate_scalar_field (const DataPostprocessorInputs::Scalar<dim> &inputs,
                                     std::vector<Vector<double> >          &computed_quantities) const
  {
    const unsigned int n_quadrature_points = static_cast<const unsigned int> (inputs.solution_gradients.size());

    Assert (computed_quantities.size() == n_quadrature_points, ExcInternalError());
    Assert (computed_quantities[0].size() == dim + 2, ExcInternalError());

    for (unsigned int q=0; q<n_quadrature_points; ++q)
      {
        for (unsigned int d=0; d<dim; ++d)
          {
            computed_quantities[q][d] = inputs.solution_gradients[q][d];
          }
        // Density
        computed_quantities[q][dim] =
          velocityPotential::internal::compute_density<dim> (inputs.solution_gradients[q],gm1,Mach_infty_square);
        // Pressure
        computed_quantities[q][dim+1] = std::pow (computed_quantities[q][dim], gas_gamma) / gas_gamma;
      }
    return;
  }

  template <int dim>
  std::vector<std::string>
  FullVelocityPotential<dim>::Postprocessor::
  get_names() const
  {
    std::vector<std::string> names;
    for (unsigned int d=0; d<dim; ++d)
      {
        names.push_back ("Velocity");
      }
    names.push_back ("Density");
    names.push_back ("Pressure");

    return names;
  }

  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  FullVelocityPotential<dim>::Postprocessor::
  get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);

    interpretation.push_back (DataComponentInterpretation::
                              component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::
                              component_is_scalar);
    return interpretation;
  }

  template <int dim>
  UpdateFlags
  FullVelocityPotential<dim>::Postprocessor::
  get_needed_update_flags() const
  {
    return (update_gradients);
  }

#include "fullVelocityPotential.inst"
}
