//
//  NSolver::compute_refinement_indicators.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II turorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


  template <int dim>
  void
  NSolver<dim>::compute_refinement_indicators()
  {
    NSVector      tmp_vector;
    tmp_vector.reinit (current_solution, true);
    tmp_vector = predictor;

    switch (parameters->refinement_indicator)
      {
      case Parameters::Refinement<dim>::Gradient:
      {

        EulerEquations<dim>::compute_refinement_indicators (dof_handler,
                                                            mapping,
                                                            tmp_vector,
                                                            refinement_indicators,
                                                            parameters->component_mask);
        break;
      }
      case Parameters::Refinement<dim>::Kelly:
      {
        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1> (3),
                                            typename FunctionMap<dim>::type(),
                                            tmp_vector,
                                            refinement_indicators,
                                            parameters->component_mask);
        break;
      }
      default:
      {
        Assert (false, ExcNotImplemented());
        break;
      }
      }
  }

#include "NSolver.inst"
}