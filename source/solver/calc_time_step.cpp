//
//  NSolver::calc_time_step.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


  // @sect4{NSolver::calc_time_step}
  //
  // Determine time step size of next time step.
  template <int dim>
  void NSolver<dim>::calc_time_step()
  {
    if (parameters->rigid_reference_time_step)
      {
        time_step = parameters->reference_time_step * CFL_number;
      }
    else
      {
        FEValues<dim> fe_v (*mapping_ptr, fe, quadrature, update_values);
        const unsigned int   n_q_points = fe_v.n_quadrature_points;
        std::vector<Vector<double> > solution_values (n_q_points,
                                                      Vector<double> (dim+2));
        double min_time_step = parameters->reference_time_step;
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
          if (cell -> is_locally_owned())
            {
              fe_v.reinit (cell);
              fe_v.get_function_values (current_solution, solution_values);

              const double cell_volumn = cell->measure();
              double face_area = 0.0;
              for (unsigned int face_no=0;
                   face_no < GeometryInfo<dim>::faces_per_cell;
                   ++face_no)
                {
                  face_area += cell->face (face_no)->measure();
                }
              const double cell_charact_length
                = static_cast<double> (GeometryInfo<dim>::faces_per_cell)
                  * cell_volumn/face_area;

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const double sound_speed
                    = EulerEquations<dim>::template compute_sound_speed (solution_values[q]);
                  const double velocity
                  = EulerEquations<dim>::template compute_velocity_magnitude (solution_values[q]);
                  min_time_step = std::min (min_time_step, cell_charact_length / velocity+sound_speed);
                }
            }
        time_step = Utilities::MPI::min (min_time_step, mpi_communicator) * CFL_number;
      }
  }

#include "NSolver.inst"
}
