//
//  main.cpp
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/print_version.h>
#include <NSolver/solver/NSolver.h>

#include <fstream>
#include <iostream>

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace NSFEMSolver;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, /* int max_num_threads */ 1);
  try
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          print_version(std::cout);
        }
      deallog.depth_console(0);

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

      bool need_try_2d = true;
      {
        // Try 3D case first because in 3D case there are more boundary and
        // initial condition components declared. Then there will be no
        // warning of 'No such entry was declared' we parsing input file.
        const unsigned int                   dimension(3);
        Parameters::AllParameters<dimension> solver_parameters;
        {
          ParameterHandler prm;

          solver_parameters.declare_parameters(prm);
          prm.parse_input(input_file);
          solver_parameters.parse_parameters(prm);
        }
        if (solver_parameters.space_dimension == dimension)
          {
            need_try_2d = false;
            NSolver<dimension> cons(&solver_parameters);
            cons.run();
          }
      }

      if (need_try_2d)
        {
          const unsigned int                   dimension(2);
          Parameters::AllParameters<dimension> solver_parameters;
          {
            ParameterHandler prm;

            solver_parameters.declare_parameters(prm);
            prm.parse_input(input_file);
            solver_parameters.parse_parameters(prm);
          }
          if (solver_parameters.space_dimension == dimension)
            {
              NSolver<dimension> cons(&solver_parameters);
              cons.run();
            }
        }
    }
  catch (std::exception &exc)
    {
      std::ofstream error_info;
      {
        std::string filename("slot");
        filename =
          filename + Utilities::int_to_string(
                       Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4);
        filename = filename + ("_runtime.error");
        error_info.open(filename.c_str());
      }
      if (!error_info)
        {
          std::cerr << "  Can not open error log file! \n"
                    << "  Aborting!" << std::endl;
          return (7);
        }
      print_version(error_info);
      error_info << std::endl
                 << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info << "Exception on processing: " << std::endl
                 << exc.what() << std::endl
                 << "Aborting!" << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info.close();
      return (2);
    }
  catch (...)
    {
      std::ofstream error_info;
      {
        std::string filename("slot");
        filename =
          filename + Utilities::int_to_string(
                       Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4);
        filename = filename + ("_runtime.error");
        error_info.open(filename.c_str());
      }
      if (!error_info)
        {
          std::cerr << "  Can not open error log file! \n"
                    << "  Aborting!" << std::endl;
          return (7);
        }
      print_version(error_info);
      error_info << std::endl
                 << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info << "Unknown exception!" << std::endl
                 << "Aborting!" << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info.close();
      return (-1);
    };
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream run_info("run.success");
      print_version(run_info);
      run_info << "Task finished successfully." << std::endl;
      run_info.close();
    }
  return (0);
}
