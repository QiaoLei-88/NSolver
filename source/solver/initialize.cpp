//
//  initialize.cpp
//  NSFEMSolver
//
//  Created by Lei Qiao on 15/8/8.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;

  template <int dim>
  void NSolver<dim>::initialize()
  {
    switch (parameters->init_method)
      {
      case Parameters::AllParameters<dim>::UserFunction :
      {
        VectorTools::interpolate (*mapping_ptr, dof_handler,
                                  parameters->initial_conditions, locally_owned_solution);
        break;
      }

      case Parameters::AllParameters<dim>::LinearVelocityPotential :
      {
        const SmartPointer<parallel::distributed::Triangulation<dim> const > triangulation_ptr (&triangulation);
        velocityPotential::LinearVelocityPotential<dim>
        linear_velocity_potential (triangulation_ptr, parameters, mpi_communicator);

        linear_velocity_potential.compute();
        linear_velocity_potential.transfer_solution (fe, dof_handler, locally_owned_solution);
        linear_velocity_potential.output_results();

        break;
      }

      case Parameters::AllParameters<dim>::FullVelocityPotential :
      {
        const SmartPointer<parallel::distributed::Triangulation<dim> const > triangulation_ptr (&triangulation);
        velocityPotential::FullVelocityPotential<dim>
        full_velocity_potential (triangulation_ptr, parameters, mpi_communicator);

        full_velocity_potential.compute();
        full_velocity_potential.transfer_solution (fe, dof_handler, locally_owned_solution);
        full_velocity_potential.output_results();

        break;
      }

      case Parameters::AllParameters<dim>::FreeStream :
      {
        std_cxx11::array<double, EquationComponents<dim>::n_components> free_stream_condition;
        // Compute free stream condition.
        // Axi definition:
        //   X in flow direction
        //   Y up
        //   Z side

        // Compute all three velocity components no matter who may space dimensions there are.
        std_cxx11::array<double,3> velocity_infty;
        velocity_infty[2] = parameters->Mach * std::sin (parameters->angle_of_side_slip);
        const double velocity_in_symm_plan =
          parameters->Mach * std::cos (parameters->angle_of_side_slip);

        velocity_infty[1] = velocity_in_symm_plan *
                            std::sin (parameters->angle_of_attack);
        velocity_infty[0] = velocity_in_symm_plan *
                            std::cos (parameters->angle_of_attack);

        for (unsigned int d=0; d<dim; ++d)
          {
            free_stream_condition[d+EquationComponents<dim>::first_velocity_component]
              =velocity_infty[d];
          }
        free_stream_condition[EquationComponents<dim>::density_component] = 1.0;
        free_stream_condition[EquationComponents<dim>::pressure_component] =
          1.0 / (parameters->gas_gamma);

        ConstantFunction<dim> free_stream_initial_value (
          &free_stream_condition[0],
          EquationComponents<dim>::n_components);

        VectorTools::interpolate (*mapping_ptr, dof_handler,
                                  free_stream_initial_value, locally_owned_solution);
        break;
      }

      default:
      {
        Assert (false, ExcNotImplemented());
        break;
      }
      }
    locally_owned_solution.compress (VectorOperation::insert);

    // Restrict strong boundary conditions
    {
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      std::vector<types::global_dof_index> dof_indices (dofs_per_cell);
      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      unsigned int cell_index (0);
      for (; cell!=endc; ++cell, ++cell_index)
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices (dof_indices);
            for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
              if (cell->at_boundary (face_no))
                {
                  const types::boundary_id boundary_id = cell->face (face_no)->boundary_id();
                  switch (parameters->boundary_conditions[boundary_id].kind)
                    {
                    case Boundary::NonSlipWall:
                    {
                      for (unsigned int i=0; i<dofs_per_cell; ++i)
                        if (fe.has_support_on_face (i, face_no) == true)
                          {
                            const unsigned int component_i = fe.system_to_component_index (i).first;
                            if (component_i != EquationComponents<dim>::density_component
                                &&
                                component_i != EquationComponents<dim>::pressure_component)
                              {
                                locally_owned_solution[dof_indices[i]] = 0.0;
                              }
                          }
                      break;
                    }
                    case Boundary::MMS_BC:
                    {
                      if (parameters->mms_use_strong_BC)
                        {
                          std::map<types::global_dof_index, double> boundary_values;
                          ComponentMask component_mask (EquationComponents<dim>::n_components, true);
                          VectorTools::interpolate_boundary_values (*mapping_ptr,
                                                                    dof_handler,
                                                                    boundary_id,
                                                                    mms,
                                                                    boundary_values,
                                                                    component_mask);
                          for (typename std::map<types::global_dof_index, double>::const_iterator
                               it = boundary_values.begin();
                               it != boundary_values.end();
                               ++it)
                            {
                              locally_owned_solution[it->first] = it->second;
                            }
                        }
                      break;
                    }
                    default:
                    {
                      break;
                    }
                    } // switch
                } // for .. face at_boundary
          } // for .. cell is_locally_owned
    }
    locally_owned_solution.compress (VectorOperation::insert);

    old_solution = locally_owned_solution;
    current_solution = old_solution;
    predictor = old_solution;

    return;
  }


#include "NSolver.inst"
}
