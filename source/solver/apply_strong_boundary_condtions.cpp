//
//  apply_strong_boundary_condtions.cpp
//  NSolver
//
//  Created by 乔磊 on 15/8/27.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#include <NSolver/solver/NSolver.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  void NSolver<dim>::apply_strong_boundary_condtions()
  {
    // For Newton method, values of strong boundary condition should be prescribed
    // in initial condition. Constraint in linear system corresponding to Newton
    // update, which always has zero values.
    std::map<types::global_dof_index, double> boundary_values;

    for (unsigned int i_boundary = 0;
         i_boundary < Parameters::AllParameters<dim>::max_n_boundaries;
         ++i_boundary)
      {
        switch (parameters->boundary_conditions[i_boundary].kind)
          {
          case Boundary::NonSlipWall:
          {
            ComponentMask component_mask (EquationComponents<dim>::n_components, true);
            // Protect density_component and pressure_component from being set to zero.
            component_mask.set (EquationComponents<dim>::density_component, false);
            component_mask.set (EquationComponents<dim>::pressure_component, false);

            VectorTools::
            interpolate_boundary_values (*mapping_ptr,
                                         dof_handler,
                                         i_boundary,
                                         ZeroFunction<dim> (EquationComponents<dim>::n_components),
                                         boundary_values,
                                         component_mask);
            break;
          }
          case Boundary::MMS_BC:
          {
            if (parameters->mms_use_strong_BC)
              {
                // Constraint all components
                ComponentMask component_mask (EquationComponents<dim>::n_components, true);
                VectorTools::
                interpolate_boundary_values (*mapping_ptr,
                                             dof_handler,
                                             i_boundary,
                                             ZeroFunction<dim> (EquationComponents<dim>::n_components),
                                             boundary_values,
                                             component_mask);
              }
            break;
          }
          default:
          {
            break;
          }
          }
      }
    // As explained in the documentation of MatrixTools::apply_boundary_values,
    // apply_boundary_values() for Trilinos never eliminate columns because this
    // operation is two expensive, and this functional is not going to be implemented.
    // However, here we are using GMRES solver and do not need the matrix to be
    // symmetric. So the parameter is set to false explicitly.
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        newton_update,
                                        right_hand_side,
                                        /*const bool  eliminate_columns = */ false);

    return;
  }

#include "NSolver.inst"
}
