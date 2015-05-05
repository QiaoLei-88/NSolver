//
//  NSEquation.cpp
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include "NSEquation.h"

namespace NSolver
{
  using namespace dealii;


  // @sect3{Euler equation specifics}

  template <int dim>
  std::vector<std::string>
  EulerEquations<dim>::component_names()
  {
    std::vector<std::string> names (dim, "velocity");
    names.push_back ("density");
    names.push_back ("pressure");

    return names;
  }


  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  EulerEquations<dim>::component_interpretation()
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


  // @sect4{Transformations between variables}

  // Next, we define the gas constant. We will set it to 1.4 in its
  // definition immediately following the declaration of this class (unlike
  // integer variables, like the ones above, static const floating point
  // member variables cannot be initialized within the class declaration in
  // C++). This value of 1.4 is representative of a gas that consists of
  // molecules composed of two atoms, such as air which consists up to small
  // traces almost entirely of $N_2$ and $O_2$.

  //static const double gas_gamma;


  // @sect4{EulerEquations::compute_refinement_indicators}

  // In this class, we also want to specify how to refine the mesh. The
  // class <code>NSolver</code> that will use all the information we
  // provide here in the <code>EulerEquation</code> class is pretty agnostic
  // about the particular conservation law it solves: as doesn't even really
  // care how many components a solution vector has. Consequently, it can't
  // know what a reasonable refinement indicator would be. On the other
  // hand, here we do, or at least we can come up with a reasonable choice:
  // we simply look at the gradient of the density, and compute
  // $\eta_K=\log\left(1+|\nabla\rho(x_K)|\right)$, where $x_K$ is the
  // center of cell $K$.
  //
  // There are certainly a number of equally reasonable refinement
  // indicators, but this one does, and it is easy to compute:
  template <int dim>
  void
  EulerEquations<dim>::compute_refinement_indicators (const DoFHandler<dim> &dof_handler,
                                                      const Mapping<dim>    &mapping,
                                                      const NSVector  &solution,
                                                      Vector<double>        &refinement_indicators)
  {
    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    std::vector<unsigned int> dofs (dofs_per_cell);

    const QMidpoint<dim>  quadrature_formula;
    const UpdateFlags update_flags = update_gradients;
    FEValues<dim> fe_v (mapping, dof_handler.get_fe(),
                        quadrature_formula, update_flags);

    std::vector<std::vector<Tensor<1,dim> > >
    dU (1, std::vector<Tensor<1,dim> > (n_components));

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
      if (cell->is_locally_owned())
        {
          fe_v.reinit (cell);
          fe_v.get_function_gradients (solution, dU);

          refinement_indicators (cell_no)
            = std::log (1+
                        std::sqrt (dU[0][density_component] *
                                   dU[0][density_component]));
        }
  }

  template <int dim>
  double EulerEquations<dim>::gas_gamma = 1.4;

  template struct EulerEquations<2>;
//  template struct EulerEquations<3>;
}
