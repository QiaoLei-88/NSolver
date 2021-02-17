#include "../RefineFunction_test.h"

#define dim 2

int main (int argc, char *argv[])
{
  typedef typename parallel::distributed::Triangulation<dim> TypeTria;
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, /* int max_num_threads */ 1);

  const unsigned int dimension (2);
  const MPI_Comm mpi_communicator = MPI_COMM_WORLD;
  const bool I_am_host = Utilities::MPI::this_mpi_process (mpi_communicator) == 0;
  const unsigned int myid = Utilities::MPI::this_mpi_process (mpi_communicator);

  // parsing input parameter
  char *input_file (0);
  char default_input_file[10] = "input.prm";

  if (argc < 2)
    {
      input_file = default_input_file;
    }
  else
    {
      input_file = argv[1];
    }

  Parameters::AllParameters<dimension> solver_parameters;
  SmartPointer<Parameters::AllParameters<dimension> const> prt_parameters (&solver_parameters);
  {
    ParameterHandler prm;

    solver_parameters.declare_parameters (prm);
    prm.parse_input (input_file);
    solver_parameters.parse_parameters (prm);
  }

  // generate initial mesh
  TypeTria triangulation (mpi_communicator);
  {
    std::vector<unsigned int> repetitions;
    repetitions.push_back (16);
    repetitions.push_back (12);

    const Point<dim> p1 (0, -1.25);
    const Point<dim> p2 (4,  1.25);

    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               repetitions,
                                               p1,
                                               p2);
  }

  std::ofstream fout;
  if (I_am_host)
    {
      fout.open ("output.out");
    }
  Vector<float> criteria;
  Vector<float> refine_mark;
  Vector<float> coarsen_mark;

  unsigned int counter = 0;
  if (I_am_host)
    {
      fout << counter << '\t'
           << triangulation.n_global_active_cells() << std::endl;
    }

  for (; counter < 51; ++counter)
    {
      // fabricate a refine criteria
      {
        const double half_PI = std::atan (1) * 2.0;
        criteria.reinit (triangulation.n_active_cells());
        typename TypeTria::active_cell_iterator
        cell = triangulation.begin_active();
        const typename TypeTria::active_cell_iterator
        endc = triangulation.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              const double x = (cell->center())[0];
              const double y = (cell->center())[1];
              const double phi = ((counter > 40)
                                  ?
                                  2.0 * half_PI
                                  :
                                  static_cast<double> (counter)/20.0 * half_PI
                                 );
              criteria[cell->active_cell_index()] =
                1.0/ (std::abs (y-std::sin (x*half_PI - phi)) + 0.01);
            }
      }

      // mark refine and coarsen
      const float threshold = 3.0;
      NSFEMSolver::Tools::refine_on_threshold (triangulation,
                                               criteria,
                                               prt_parameters,
                                               threshold);
      triangulation.prepare_coarsening_and_refinement();

      // Save refine and coarsen marks
      {
        refine_mark.reinit (triangulation.n_active_cells());
        coarsen_mark.reinit (triangulation.n_active_cells());

        typename TypeTria::active_cell_iterator
        cell = triangulation.begin_active();
        const typename TypeTria::active_cell_iterator
        endc = triangulation.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              refine_mark[cell->active_cell_index()] =
                (cell->refine_flag_set()
                 ?
                 1.0
                 :
                 0.0);
              coarsen_mark[cell->active_cell_index()] =
                (cell->coarsen_flag_set()
                 ?
                 1.0
                 :
                 0.0);
            }
      }

      // Write out visualization data
      {
        DataOut<dim> data_out;
        data_out.attach_triangulation (triangulation);
        {
          const std::string data_name ("indicator");
          data_out.add_data_vector (criteria,
                                    data_name,
                                    DataOut<dim>::type_cell_data);
        }
        {
          const std::string data_name ("refine_flag");
          data_out.add_data_vector (refine_mark,
                                    data_name,
                                    DataOut<dim>::type_cell_data);
        }
        {
          const std::string data_name ("coarsen_flag");
          data_out.add_data_vector (coarsen_mark,
                                    data_name,
                                    DataOut<dim>::type_cell_data);
        }

        Vector<float> subdomain (triangulation.n_active_cells());
        {
          std::fill (subdomain.begin(), subdomain.end(), myid);
          const std::string data_name ("sub_domain");
          data_out.add_data_vector (subdomain,
                                    data_name,
                                    DataOut<dim>::type_cell_data);
        }
        data_out.build_patches();

        const std::string output_tag = "grid-" +
                                       Utilities::int_to_string (counter, 4);
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
      }

      triangulation.execute_coarsening_and_refinement();

      if (I_am_host)
        {
          fout << counter << '\t'
               << triangulation.n_global_active_cells() << std::endl;
        }
    };
  fout.close();
  return (0);
}
