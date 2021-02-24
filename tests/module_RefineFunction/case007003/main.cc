#include "../RefineFunction_test.h"

#define dim 2

int
main(int argc, char *argv[])
{
  typedef typename parallel::distributed::Triangulation<dim> TypeTria;
  Utilities::MPI::MPI_InitFinalize                           mpi_initialization(
    argc, argv, /* int max_num_threads */ 1);

  const unsigned int dimension(2);
  const MPI_Comm     mpi_communicator = MPI_COMM_WORLD;
  const bool         I_am_host =
    Utilities::MPI::this_mpi_process(mpi_communicator) == 0;
  const unsigned int myid = Utilities::MPI::this_mpi_process(mpi_communicator);

  // parsing input parameter
  char *input_file(0);
  char  default_input_file[10] = "input.prm";

  if (argc < 2)
    {
      input_file = default_input_file;
    }
  else
    {
      input_file = argv[1];
    }

  Parameters::AllParameters<dimension>                     solver_parameters;
  SmartPointer<Parameters::AllParameters<dimension> const> prt_parameters(
    &solver_parameters);
  {
    ParameterHandler prm;

    solver_parameters.declare_parameters(prm);
    prm.parse_input(input_file);
    solver_parameters.parse_parameters(prm);
  }

  // generate initial mesh
  TypeTria triangulation(
    mpi_communicator,
    Triangulation<dim>::none,
    parallel::distributed::Triangulation<dim>::no_automatic_repartitioning);
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);
  }

  std::ofstream fout;
  if (I_am_host)
    {
      fout.open("output.out");
    }
  Vector<float> criteria;
  Vector<float> refine_mark;
  Vector<float> coarsen_mark;

  unsigned int counter = 0;
  if (I_am_host)
    {
      fout << counter << '\t' << triangulation.n_global_active_cells()
           << std::endl;
    }

  while (counter < prt_parameters->max_refine_level)
    {
      // fabricate a refine criteria
      {
        criteria.reinit(triangulation.n_active_cells());
        typename TypeTria::active_cell_iterator cell =
          triangulation.begin_active();
        const typename TypeTria::active_cell_iterator endc =
          triangulation.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              criteria[cell->active_cell_index()] =
                ((cell->center()).norm() + 0.1);
            }
      }

      // mark refine and coarsen
      NSFEMSolver::Tools::refine_and_coarsen_fixed_number(triangulation,
                                                          criteria,
                                                          prt_parameters);
      triangulation.prepare_coarsening_and_refinement();

      {
        refine_mark.reinit(triangulation.n_active_cells());
        coarsen_mark.reinit(triangulation.n_active_cells());

        typename TypeTria::active_cell_iterator cell =
          triangulation.begin_active();
        const typename TypeTria::active_cell_iterator endc =
          triangulation.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              refine_mark[cell->active_cell_index()] =
                (cell->refine_flag_set() ? 1.0 : 0.0);
              coarsen_mark[cell->active_cell_index()] =
                (cell->coarsen_flag_set() ? 1.0 : 0.0);
            }
      }

      {
        DataOut<dim> data_out;
        data_out.attach_triangulation(triangulation);
        {
          const std::string data_name("indicator");
          data_out.add_data_vector(criteria,
                                   data_name,
                                   DataOut<dim>::type_cell_data);
        }
        {
          const std::string data_name("refine_flag");
          data_out.add_data_vector(refine_mark,
                                   data_name,
                                   DataOut<dim>::type_cell_data);
        }
        {
          const std::string data_name("coarsen_flag");
          data_out.add_data_vector(coarsen_mark,
                                   data_name,
                                   DataOut<dim>::type_cell_data);
        }

        data_out.build_patches();

        const std::string output_tag =
          "grid-" + Utilities::int_to_string(counter, 4);
        const std::string slot_itag =
          ".slot-" + Utilities::int_to_string(myid, 4);

        std::ofstream output((output_tag + slot_itag + ".vtu").c_str());
        data_out.write_vtu(output);

        if (I_am_host)
          {
            std::vector<std::string> filenames;
            for (unsigned int i = 0;
                 i < Utilities::MPI::n_mpi_processes(mpi_communicator);
                 ++i)
              {
                filenames.push_back(output_tag + ".slot-" +
                                    Utilities::int_to_string(i, 4) + ".vtu");
              }
            std::ofstream master_output((output_tag + ".pvtu").c_str());
            data_out.write_pvtu_record(master_output, filenames);
          }
      }

      triangulation.execute_coarsening_and_refinement();
      ++counter;


      if (I_am_host)
        {
          fout << counter << '\t' << triangulation.n_global_active_cells()
               << std::endl;
        }
    };
  fout.close();
  return (0);
}
