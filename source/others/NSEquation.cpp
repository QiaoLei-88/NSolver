//
//  NSEquation.cpp
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/NSEquation.h>

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  void EulerEquations<dim>::set_parameter (const Parameters::AllParameters<dim> *const para_ptr_in)
  {
    Assert (para_ptr_in, ExcMessage ("Null pointer encountered!"));
    parameters = para_ptr_in;
  }

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
  EulerEquations<dim>::compute_refinement_indicators (DoFHandler<dim> const &dof_handler,
                                                      Mapping<dim>    const &mapping,
                                                      NSVector        const &solution,
                                                      Vector<float>         &refinement_indicators,
                                                      ComponentMask   const &component_mask)
  {
    Assert (component_mask.size() >= n_components, ExcMessage ("component_mask doesn't have enough elements"));

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

          refinement_indicators[cell_no] = 0.0;
          for (unsigned int ic=0; ic<n_components; ++ic)
            {
              if (component_mask[ic])
                {
                  refinement_indicators[cell_no] +=
                    static_cast<float> (std::log (1.0+std::sqrt (dU[0][ic]*dU[0][ic])));
                }
            }
        }
  }

  template <int dim>
  double EulerEquations<dim>::gas_gamma = 1.4;

  template <int dim>
  Parameters::AllParameters<dim> const *
  EulerEquations<dim>::parameters = 0;

  template class EulerEquations<2>;
//  template struct EulerEquations<3>;
}
