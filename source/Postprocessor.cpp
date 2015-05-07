//
//  Created by 乔磊 on 15/4/24.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include "Postprocessor.h"


namespace NSolver
{
  using namespace dealii;


  template <int dim>
  Postprocessor<dim>::
  Postprocessor (Parameters::AllParameters<dim>const *const para_ptr_in,
                 MMS const *const mms_ptr_in)
    :
    para_ptr (para_ptr_in),
    mms_ptr (mms_ptr_in),
    do_schlieren_plot (para_ptr_in->schlieren_plot),
    output_mms (para_ptr_in->n_mms == 1)
  {}


  // This is the only function worth commenting on. When generating graphical
  // output, the DataOut and related classes will call this function on each
  // cell, with values, gradients, Hessians, and normal vectors (in case we're
  // working on faces) at each quadrature point. Note that the data at each
  // quadrature point is itself vector-valued, namely the conserved
  // variables. What we're going to do here is to compute the quantities we're
  // interested in at each quadrature point. Note that for this we can ignore
  // the Hessians ("dduh") and normal vectors; to avoid compiler warnings
  // about unused variables, we comment out their names.
  template <int dim>
  void
  Postprocessor<dim>::
  compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                     const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                     const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                     const std::vector<Point<dim> >                  &/*normals*/,
                                     const std::vector<Point<dim> >                  &points,
                                     std::vector<Vector<double> >                    &computed_quantities) const
  {
    // At the beginning of the function, let us make sure that all variables
    // have the correct sizes, so that we can access individual vector
    // elements without having to wonder whether we might read or write
    // invalid elements; we also check that the <code>duh</code> vector only
    // contains data if we really need it (the system knows about this because
    // we say so in the <code>get_needed_update_flags()</code> function
    // below). For the inner vectors, we check that at least the first element
    // of the outer vector has the correct inner size:
    const unsigned int n_quadrature_points = static_cast<const unsigned int> (uh.size());

    if (do_schlieren_plot)
      {
        Assert (duh.size() == n_quadrature_points,ExcInternalError());
      }
    else
      {
        Assert (duh.size() == 0,ExcInternalError());
      }

    Assert (computed_quantities.size() == n_quadrature_points,ExcInternalError());
    Assert (uh[0].size() == EulerEquations<dim>::n_components,ExcInternalError());

    //MMS: Extra memmory space
    Vector<double>::size_type expected_size = dim+1;
    if (do_schlieren_plot)
      {
        expected_size += 1;
      }
    if (output_mms)
      {
        expected_size += 3*EulerEquations<dim>::n_components;
      }
    Assert (computed_quantities[0].size() == expected_size, ExcInternalError());

    // Then loop over all quadrature points and do our work there. The code
    // should be pretty self-explanatory. The order of output variables is
    // first <code>dim</code> momentums, then the energy_density, and if so desired
    // the schlieren plot. Note that we try to be generic about the order of
    // variables in the input vector, using the
    // <code>first_momentum_component</code> and
    // <code>density_component</code> information:
    for (unsigned int q=0; q<n_quadrature_points; ++q)
      {
        const double density = uh[q] (EulerEquations<dim>::density_component);

        for (unsigned int d=0; d<dim; ++d)
          computed_quantities[q] (d)
            = uh[q] (EulerEquations<dim>::first_momentum_component+d) * density;

        computed_quantities[q] (dim) = EulerEquations<dim>::compute_energy_density (uh[q]);

        if (do_schlieren_plot)
          computed_quantities[q] (dim+1) = duh[q][EulerEquations<dim>::density_component] *
                                           duh[q][EulerEquations<dim>::density_component];

        if (output_mms)
          {
            std_cxx11::array<double, EulerEquations<dim>::n_components> sol, src;
            mms_x.evaluate (points[q],sol,src,true);
            int i_out = dim + 2;
            for (unsigned int ic = 0; ic < EulerEquations<dim>::n_components; ++ic, ++i_out)
              {
                computed_quantities[q][i_out] = src[ic];
              }
            for (unsigned int ic = 0; ic < EulerEquations<dim>::n_components; ++ic, ++i_out)
              {
                computed_quantities[q][i_out] = sol[ic];
              }
            for (unsigned int ic = 0; ic < EulerEquations<dim>::n_components; ++ic, ++i_out)
              {
                computed_quantities[q][i_out] = sol[ic] - uh[q][ic];
              }
          } // End if (output_mms)
      }
  }


  template <int dim>
  std::vector<std::string>
  Postprocessor<dim>::
  get_names() const
  {
    std::vector<std::string> names;
    for (unsigned int d=0; d<dim; ++d)
      {
        names.push_back ("momentum");
      }
    names.push_back ("energy_density");

    if (do_schlieren_plot)
      {
        names.push_back ("schlieren_plot");
      }

    //MMS: Exact output
    if (output_mms)
      {

        for (unsigned int d=0; d<dim; ++d)
          {
            names.push_back ("mms_src_momentum");
          }
        names.push_back ("mms_src_density");
        names.push_back ("mms_src_energy");
        for (unsigned int d=0; d<dim; ++d)
          {
            names.push_back ("mms_exact_velocity");
          }
        names.push_back ("mms_exact_density");
        names.push_back ("mms_exact_energy");
//    for (unsigned int d=0; d<dim; ++d)
//      {
        names.push_back ("mms_error_velocity_x");
        names.push_back ("mms_error_velocity_y");
//      }
        names.push_back ("mms_error_density");
        names.push_back ("mms_error_energy");
      }// End MMS: Exact output

    return names;
  }


  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  Postprocessor<dim>::
  get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);

    interpretation.push_back (DataComponentInterpretation::
                              component_is_scalar);

    if (do_schlieren_plot)
      interpretation.push_back (DataComponentInterpretation::
                                component_is_scalar);

    // MMS:
    if (output_mms)
      {
        for (unsigned int d=0; d<dim; ++d)
          {
            interpretation.push_back (DataComponentInterpretation::
                                      component_is_part_of_vector);
          }
        interpretation.push_back (DataComponentInterpretation::
                                  component_is_scalar);
        interpretation.push_back (DataComponentInterpretation::
                                  component_is_scalar);

        for (unsigned int d=0; d<dim; ++d)
          {
            interpretation.push_back (DataComponentInterpretation::
                                      component_is_part_of_vector);
          }
        interpretation.push_back (DataComponentInterpretation::
                                  component_is_scalar);
        interpretation.push_back (DataComponentInterpretation::
                                  component_is_scalar);

        for (unsigned int d=0; d<dim; ++d)
          {
            interpretation.push_back (DataComponentInterpretation::
                                      component_is_scalar);
          }
        interpretation.push_back (DataComponentInterpretation::
                                  component_is_scalar);
        interpretation.push_back (DataComponentInterpretation::
                                  component_is_scalar);
      }// END MMS:
    return interpretation;
  }



  template <int dim>
  UpdateFlags
  Postprocessor<dim>::
  get_needed_update_flags() const
  {
    UpdateFlags flags = update_values;
    // MMS : update_quadrature_points
    if (do_schlieren_plot)
      {
        flags |= update_gradients;
      }
    if (output_mms)
      {
        flags |= update_quadrature_points;
      }
    return (flags);
  }

  template class Postprocessor<2>;
}
