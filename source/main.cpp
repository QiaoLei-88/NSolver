//
//  main.cpp
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "ConservationLaw.h"


int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace Step33;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, /* int max_num_threads */ 1);
      ConditionalOStream pcout(std::cout,
                               (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      deallog.depth_console(0);
      if (argc != 2)
        {
          pcout << "Usage:" << argv[0] << " input_file" << std::endl;
          std::exit(1);
        }

      ConservationLaw<2> cons (argv[1]);
      cons.run ();
    }
  catch (std::exception &exc)
    {
      std::ofstream error_info;
      {
        std::string filename("node");
        filename = filename + Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4);
        filename = filename + ("_runtime.error");
        error_info.open(filename.c_str());
      }
      if (!error_info)
        {
          std::cerr << "  Can not open error log file! \n"
                    << "  Aborting!" << std::endl;
          return (1);
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
        std::string filename("node");
        filename = filename + Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4);
        filename = filename + ("_runtime.error");
        error_info.open(filename.c_str());
      }
      if (!error_info)
        {
          std::cerr << "  Can not open error log file! \n"
                    << "  Aborting!" << std::endl;
          return (1);
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

  std::ofstream run_info("run.success");
  run_info << "Task finished successfully." << std::endl;
  run_info.close();
  return (0);
}