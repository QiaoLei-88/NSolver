//  Created by 乔磊 on 2015/8/7.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/linearVelocityPotential/linearVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void
  LinearVelocityPotential<dim>::Postprocessor::
  compute_derived_quantities_scalar (const std::vector<double>             &uh,
                                     const std::vector<Tensor<1,dim> >     &duh,
                                     const std::vector<Tensor<2,dim> >     &/*dduh*/,
                                     const std::vector<Point<dim> >        &/*normals*/,
                                     const std::vector<Point<dim> >        &/*points*/,
                                     std::vector<Vector<double> >          &computed_quantities) const
  {
    const unsigned int n_quadrature_points = static_cast<const unsigned int> (uh.size());

    Assert (duh.size() == n_quadrature_points, ExcInternalError());
    Assert (computed_quantities.size() == n_quadrature_points, ExcInternalError());
    Assert (computed_quantities[0].size() == dim + 2, ExcInternalError());

    for (unsigned int q=0; q<n_quadrature_points; ++q)
      {
        double Mach_local = 0;
        for (unsigned int d=0; d<dim; ++d)
          {
            computed_quantities[q][d] = velocity_infty[d] + duh[q][d];
            Mach_local += computed_quantities[q][d] * computed_quantities[q][d];
          }

        const double gas_gamma (1.4);
        const double Mach_infty (velocity_infty.square());
        const double Mach_ratio =
          (1.0 + 0.5* (gas_gamma-1.0) * Mach_infty) /
          (1.0 + 0.5* (gas_gamma-1.0) * Mach_local);
        computed_quantities[q][dim]   = std::pow (Mach_ratio, 1.0/ (gas_gamma - 1.0));
        computed_quantities[q][dim+1] = std::pow (Mach_ratio, gas_gamma/ (gas_gamma - 1.0)) / gas_gamma;
      }
    return;
  }

  template <int dim>
  std::vector<std::string>
  LinearVelocityPotential<dim>::Postprocessor::
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
  LinearVelocityPotential<dim>::Postprocessor::
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
  LinearVelocityPotential<dim>::Postprocessor::
  get_needed_update_flags() const
  {
    return (update_values | update_gradients);
  }

#include "linearVelocityPotential.inst.in"
}
