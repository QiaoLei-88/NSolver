//
//  NSolver::check_negative_density_pressure.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  void NSolver<dim>::check_negative_density_pressure() const
  {
    FEValues<dim> fe_v (*mapping_ptr, fe, quadrature, update_values);
    const unsigned int   n_q_points = fe_v.n_quadrature_points;
    std::vector<Vector<double> > solution_values (n_q_points,
                                                  Vector<double> (EquationComponents<dim>::n_components));
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell -> is_locally_owned())
        {
          fe_v.reinit (cell);
          fe_v.get_function_values (current_solution, solution_values);
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const double density = solution_values[q] (EquationComponents<dim>::density_component);
              AssertThrow (density > 0.0, ExcMessage ("Negative density encountered!"));
              const double pressure = solution_values[q] (EquationComponents<dim>::pressure_component);
              AssertThrow (pressure > 0.0, ExcMessage ("Negative pressure encountered!"));
            }
        }
  }

#include "NSolver.inst"
}
