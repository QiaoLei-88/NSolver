
//  Created by 乔磊 on 2015/9/8.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/velocityPotential/fullVelocityPotential.h>

namespace velocityPotential
{
  template <int dim>
  void FullVelocityPotential<dim>::output_results() const
  {
    FullVelocityPotential<dim>::Postprocessor postprocessor (Mach_infty_square, gas_gamma);;
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution, "velocityPotential");

    Vector<float> subdomain (triangulation->n_active_cells());
    std::fill (subdomain.begin(), subdomain.end(),triangulation->locally_owned_subdomain());
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.add_data_vector (locally_relevant_solution, postprocessor);

    data_out.build_patches();

    const std::string filename = ("fullVelocityPotentialSolution." +
                                  Utilities::int_to_string
                                  (triangulation->locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process (mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes (mpi_communicator);
             ++i)
          filenames.push_back ("fullVelocityPotentialSolution." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");

        std::ofstream master_output ("fullVelocityPotentialSolution.pvtu");
        data_out.write_pvtu_record (master_output, filenames);
      }
  }

#include "fullVelocityPotential.inst"
}
