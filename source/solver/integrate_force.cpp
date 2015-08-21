//
//  NSolver_Forceintegrate_force.cpp
//  NSolver
//
//  Created by 乔磊 on 15/5/10.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//


#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  void NSolver<dim>::integrate_force (Parameters::AllParameters<dim> const *const parameters,
                                      WallForce &wall_force) const
  {
    const UpdateFlags face_update_flags = update_values
                                          | update_quadrature_points
                                          | update_JxW_values
                                          | update_normal_vectors;

    FEFaceValues<dim> fe_v_face (*mapping_ptr, fe, face_quadrature,face_update_flags);
    wall_force.clear();

    std::vector<Vector<double> > solution_values;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int cell_index (0);
    for (; cell!=endc; ++cell, ++cell_index)
      if (cell->is_locally_owned())
        for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          if (cell->at_boundary (face_no))
            if (parameters->sum_force[cell->face (face_no)->boundary_id()])
              {
                fe_v_face.reinit (cell, face_no);
                unsigned int const n_q_points = fe_v_face.n_quadrature_points;
                solution_values.resize (n_q_points,
                                        Vector<double> (EquationComponents<dim>::n_components));
                fe_v_face.get_function_values (current_solution, solution_values);

                for (unsigned int q=0; q<n_q_points; ++q)
                  {
                    Point<dim> const wall_norm = fe_v_face.normal_vector (q);
                    Point<dim> const wall_position = fe_v_face.quadrature_point (q);
                    double f_ds[3];
                    double moment_arm[3] = {0.,0.,0.};

                    for (unsigned int id=0; id<dim; ++id)
                      {
                        f_ds[id] = solution_values[q][EquationComponents<dim>::pressure_component] *
                                   wall_norm[id] * fe_v_face.JxW (q);
                        wall_force.force[id] += f_ds[id];
                        moment_arm[id] = wall_position[id] - parameters->moment_center[id];
                      }

                    {
                      unsigned int const x=0;
                      unsigned int const y=1;
                      unsigned int const z=2;

                      wall_force.moment[x] -= f_ds[y]*moment_arm[z];
                      wall_force.moment[x] += f_ds[z]*moment_arm[y];

                      wall_force.moment[y] -= f_ds[z]*moment_arm[x];
                      wall_force.moment[y] += f_ds[x]*moment_arm[z];

                      wall_force.moment[z] -= f_ds[x]*moment_arm[y];
                      wall_force.moment[z] += f_ds[y]*moment_arm[x];
                    }

                  }
              }
    wall_force.mpi_sum (mpi_communicator);
    // Turn force into force coefficient
    double const force_to_coeff = 0.5 * parameters->Mach *
                                  parameters->Mach * parameters->reference_area;
    wall_force.force[0] /= force_to_coeff;
    wall_force.force[1] /= force_to_coeff;
    wall_force.force[2] /= force_to_coeff;

    // Roll moment coefficient
    wall_force.moment[0] /= force_to_coeff;
    wall_force.moment[0] /= parameters->reference_span;
    // Yaw mement coefficient
    wall_force.moment[1] /= force_to_coeff;
    wall_force.moment[1] /= parameters->reference_span;

    // Pitch momernt coefficient
    wall_force.moment[2] /= force_to_coeff;
    wall_force.moment[2] /= parameters->reference_chord;

    // Project force coefficient to lift and drag component
    double const sin_aoa = std::sin (parameters->angle_of_attack);
    double const cos_aoa = std::cos (parameters->angle_of_attack);
    wall_force.lift = cos_aoa*wall_force.force[1] - sin_aoa*wall_force.force[0];
    wall_force.drag = cos_aoa*wall_force.force[0] + sin_aoa*wall_force.force[1];
  }

#include "NSolver.inst"
}
