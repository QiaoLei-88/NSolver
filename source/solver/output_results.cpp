//
//  NSolver::output_results.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


  // @sect4{NSolver::output_results}

  // This function now is rather straightforward. All the magic, including
  // transforming data from conservative variables to physical ones has been
  // abstracted and moved into the EulerEquations class so that it can be
  // replaced in case we want to solve some other hyperbolic conservation law.
  //
  // Note that the number of the output file is determined by keeping a
  // counter in the form of a static variable that is set to zero the first
  // time we come to this function and is incremented by one at the end of
  // each invocation.
  template <int dim>
  void NSolver<dim>::output_results() const
  {
    Postprocessor<dim> postprocessor (parameters, &mms);

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);

    data_out.add_data_vector (current_solution,
                              EquationComponents<dim>::component_names(),
                              DataOut<dim>::type_dof_data,
                              EquationComponents<dim>::component_interpretation());

    data_out.add_data_vector (current_solution, postprocessor);

    {
      const std::string data_name ("artificial_viscosity");
      data_out.add_data_vector (artificial_viscosity,
                                data_name,
                                DataOut<dim>::type_cell_data);
    }
    {
      const std::string data_name ("refinement_indicators");
      data_out.add_data_vector (refinement_indicators,
                                data_name,
                                DataOut<dim>::type_cell_data);
    }
    {
      const std::string data_name ("local_time_step_size");
      data_out.add_data_vector (local_time_step_size,
                                data_name,
                                DataOut<dim>::type_cell_data);
    }
    {
      const std::string data_name ("laplacian_indicator");
      data_out.add_data_vector (laplacian_indicator,
                                data_name,
                                DataOut<dim>::type_cell_data);
    }
    {
      AssertThrow (dim <= 3, ExcNotImplemented());
      char prefix[3][4] = {"1st", "2nd", "3rd"};

      std::vector<std::string> data_names;
      data_names.clear();
      for (unsigned id=0; id < dim; ++id)
        {
          data_names.push_back ("_residualAt1stNewtonIter");
          data_names[id] = prefix[id] + data_names[id];
          data_names[id] = "momentum" + data_names[id];
        }

      data_names.push_back ("density_residualAt1stNewtonIter");
      data_names.push_back ("energy_residualAt1stNewtonIter");

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim+2, DataComponentInterpretation::component_is_scalar);

      data_out.add_data_vector (residual_for_output,
                                data_names,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);
    }

    Vector<float> subdomain (triangulation.n_active_cells());
    std::fill (subdomain.begin(), subdomain.end(), myid);
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches();


    const std::string output_tag = "solution-" +
                                   Utilities::int_to_string (field_output_counter, 4);
    const std::string slot_itag = ".slot-" + Utilities::int_to_string (myid, 4);

    std::ofstream output ((output_tag + slot_itag + ".vtu").c_str());
    data_out.write_vtu (output);

    if (I_am_host)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes (mpi_communicator);
             ++i)
          {
            filenames.push_back (output_tag +
                                 ".slot-" +
                                 Utilities::int_to_string (i, 4) +
                                 ".vtu");
          }
        std::ofstream master_output ((output_tag + ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }

    ++field_output_counter;
  }

#include "NSolver.inst"
}
