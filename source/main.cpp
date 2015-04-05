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
  std::ofstream error_info("runtime.error");
  if (!error_info)
    {
      std::cerr << "  Can not open error log file! \n"
                << "  Aborting!" << std::endl;
      return (1);
    }
  try
    {
      using namespace dealii;
      using namespace Step33;

      deallog.depth_console(0);
      if (argc != 2)
        {
          std::cout << "Usage:" << argv[0] << " input_file" << std::endl;
          std::exit(1);
        }

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, dealii::numbers::invalid_unsigned_int);

      ConservationLaw<2> cons (argv[1]);
      cons.run ();
    }
  catch (std::exception &exc)
    {
      error_info << std::endl << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info << "Exception on processing: " << std::endl
                 << exc.what() << std::endl
                 << "Aborting!" << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      return (2);
    }
  catch (...)
    {
      error_info << std::endl << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      error_info << "Unknown exception!" << std::endl
                 << "Aborting!" << std::endl
                 << "----------------------------------------------------"
                 << std::endl;
      return (-1);
    };

  return (0);
}