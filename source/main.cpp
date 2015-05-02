//
//  main.cpp
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "NSolver.h"


int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace NSolver;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, /* int max_num_threads */ 1);

      deallog.depth_console (0);

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

      Parameters::FEParameters fe_parameters;
      {
        ParameterHandler prm;
        const ParameterHandler::ReadInputFlags flags =
          ParameterHandler::ReadInputFlag_IgnoreUndeclaredEntry
          | ParameterHandler::ReadInputFlag_InputFileHelp;
        fe_parameters.declare_parameters (prm);
        prm.read_input (input_file, flags);
        fe_parameters.parse_parameters (prm);
      }

      NSolver<2> cons (input_file, fe_parameters);
      cons.run();
    }
  catch (std::exception &exc)
    {
      std::ofstream error_info;
      {
        std::string filename ("slot");
        filename = filename + Utilities::int_to_string (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD), 4);
        filename = filename + ("_runtime.error");
        error_info.open (filename.c_str());
      }
      if (!error_info)
        {
          std::cerr << "  Can not open error log file! \n"
                    << "  Aborting!" << std::endl;
          return (7);
        }

      error_info << std::endl << std::endl
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
        std::string filename ("slot");
        filename = filename + Utilities::int_to_string (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD), 4);
        filename = filename + ("_runtime.error");
        error_info.open (filename.c_str());
      }
      if (!error_info)
        {
          std::cerr << "  Can not open error log file! \n"
                    << "  Aborting!" << std::endl;
          return (7);
        }

      error_info << std::endl << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info << "Unknown exception!" << std::endl
                 << "Aborting!" << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info.close();
      return (-1);
    };

  std::ofstream run_info ("run.success");
  run_info << "Task finished successfully." << std::endl;
  run_info.close();
  return (0);
}