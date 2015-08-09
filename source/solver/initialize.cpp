//
//  initialize.cpp
//  NSFEMSolver
//
//  Created by Lei Qiao on 15/8/8.
//  A work based on deal.II turorial step-33.
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
        VectorTools::interpolate (dof_handler,
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
          1.0/EulerEquations<dim>::gas_gamma;

        ConstantVectorFunction<dim> free_stream_initial_value (
          &free_stream_condition[0],
          EquationComponents<dim>::n_components);

        VectorTools::interpolate (mapping, dof_handler,
                                  free_stream_initial_value, locally_owned_solution);
        break;
      }

      default:
      {
        Assert (false, ExcNotImplemented());
        break;
      }
      }

    old_solution = locally_owned_solution;
    current_solution = old_solution;
    predictor = old_solution;

    return;
  }


  template class NSolver<2>;
}
