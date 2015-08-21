//
//  NSFEMSolver::NSolver.cpp
//  NSFEMSolver
//
//  Created by Lei Qiao on 15/2/3.
//  A work based on deal.II tutorial step-33.
//

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2007 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: David Neckels, Boulder, Colorado, 2007, 2008
 */


#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;

  // @sect3{Conservation law class}

  // Here finally comes the class that actually does something with all the
  // Euler equation and parameter specifics we've defined above. The public
  // interface is pretty much the same as always (the constructor now takes
  // the name of a file from which to read parameters, which is passed on the
  // command line). The private function interface is also pretty similar to
  // the usual arrangement, with the <code>assemble_system</code> function
  // split into three parts: one that contains the main loop over all cells
  // and that then calls the other two for integrals over cells and faces,
  // respectively.


  // @sect4{NSolver::NSolver}
  //
  // There is nothing much to say about the constructor. Essentially, it reads
  // the input file and fills the parameter object with the parsed values:
  template <int dim>
  NSolver<dim>::NSolver (Parameters::AllParameters<dim> *const para_ptr_in)
    :
    mpi_communicator (MPI_COMM_WORLD),
    spherical_boundary (Point<dim>()/*=(0,0,...)*/,/*radius=*/0.5),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    mapping_ptr (0),
    fe (FE_Q<dim> (para_ptr_in->fe_degree), EquationComponents<dim>::n_components),
    dof_handler (triangulation),
    quadrature (para_ptr_in->quadrature_degree),
    face_quadrature (para_ptr_in->face_quadrature_degree),
    I_am_host (Utilities::MPI::this_mpi_process (mpi_communicator) == 0),
    myid (Utilities::MPI::this_mpi_process (mpi_communicator)),
    parameters (para_ptr_in),
    verbose_cout (std::cout, false),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process (mpi_communicator)
            == 0)),
    computing_timer (MPI_COMM_WORLD,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::cpu_and_wall_times),
    CFL_number (para_ptr_in->CFL_number),
    n_sparsity_pattern_out (-1)
  {
    Assert (parameters, ExcMessage ("Null pointer encountered!"));

    switch (parameters->mapping_type)
      {
      case Parameters::FEParameters::MappingQ:
      {
        if (parameters->mapping_degree == 1)
          {
            mapping_ptr = new MappingQ1<dim>();
          }
        else
          {
            mapping_ptr = new MappingQ<dim> (parameters->mapping_degree);
          }
        break;
      }
      case Parameters::FEParameters::MappingC1:
      {
        mapping_ptr = new MappingC1<dim>();
        break;
      }
      default:
      {
        AssertThrow (false, ExcNotImplemented());
        break;
      }
      }

    EulerEquations<dim>::gas_gamma = 1.4;
    EulerEquations<dim>::set_parameter (parameters);

    verbose_cout.set_condition (parameters->output == Parameters::Solver::verbose);

    if (parameters->n_mms == 1)
      {
        // Setup coefficients for MMS
        std_cxx11::array<Coeff_2D, EquationComponents<dim>::n_components> coeffs;
        // // component u:
        coeffs[0].c0  = 0.2;
        coeffs[0].cx  = 0.01;
        coeffs[0].cy  = -0.02;
        coeffs[0].cxy = 0;
        coeffs[0].ax  = 1.5;
        coeffs[0].ay  = 0.6;
        coeffs[0].axy = 0;

        // component v:
        coeffs[1].c0  = 0.4;
        coeffs[1].cx  = -0.01;
        coeffs[1].cy  = 0.04;
        coeffs[1].cxy = 0;
        coeffs[1].ax  = 0.5;
        coeffs[1].ay  = 2.0/3.0;
        coeffs[1].axy = 0;

        // component density:
        coeffs[2].c0  = 1.0;
        coeffs[2].cx  = 0.15;
        coeffs[2].cy  = -0.1;
        coeffs[2].cxy = 0;
        coeffs[2].ax  = 1.0;
        coeffs[2].ay  = 0.5;
        coeffs[2].axy = 0;

        // component pressure:
        coeffs[3].c0  = 1.0/1.4;
        coeffs[3].cx  = 0.2;
        coeffs[3].cy  = 0.5;
        coeffs[3].cxy = 0;
        coeffs[3].ax  = 2.0;
        coeffs[3].ay  = 1.0;
        coeffs[3].axy = 0;
        // component u:
        // coeffs[0].c0  = 2.0;
        // coeffs[0].cx  = 0.2;
        // coeffs[0].cy  = -0.1;
        // coeffs[0].cxy = 0;
        // coeffs[0].ax  = 1.5;
        // coeffs[0].ay  = 0.6;
        // coeffs[0].axy = 0;

        // // component v:
        // coeffs[1].c0  = 2.0;
        // coeffs[1].cx  = -0.25;
        // coeffs[1].cy  = 0.125;
        // coeffs[1].cxy = 0;
        // coeffs[1].ax  = 0.5;
        // coeffs[1].ay  = 2.0/3.0;
        // coeffs[1].axy = 0;

        // // component density:
        // coeffs[2].c0  = 1.0;
        // coeffs[2].cx  = 0.15;
        // coeffs[2].cy  = -0.1;
        // coeffs[2].cxy = 0;
        // coeffs[2].ax  = 1.0;
        // coeffs[2].ay  = 0.5;
        // coeffs[2].axy = 0;

        // // component pressure:
        // coeffs[3].c0  = 0.72;
        // coeffs[3].cx  = 0.2;
        // coeffs[3].cy  = 0.5;
        // coeffs[3].cxy = 0;
        // coeffs[3].ax  = 2.0;
        // coeffs[3].ay  = 1.0;
        // coeffs[3].axy = 0;

        // Initialize MMS
        mms.reinit (coeffs);
      }
  }

  template <int dim>
  NSolver<dim>::~NSolver()
  {
    delete mapping_ptr;
  }

#include "NSolver.inst"
}

