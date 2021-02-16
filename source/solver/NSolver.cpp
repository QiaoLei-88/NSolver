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
    spherical_boundary (0),
    NACA_foil_boundary (para_ptr_in->NACA_foil, 1.0),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    mapping_ptr (0),
    fe (FE_Q<dim> (para_ptr_in->fe_degree), EquationComponents<dim>::n_components),
    dof_handler (triangulation),
    quadrature (para_ptr_in->quadrature_degree),
    face_quadrature (para_ptr_in->face_quadrature_degree),
    blend_artificial_viscosity (false),
    I_am_host (Utilities::MPI::this_mpi_process (mpi_communicator) == 0),
    myid (Utilities::MPI::this_mpi_process (mpi_communicator)),
    parameters (para_ptr_in),
    parameters_modifier (para_ptr_in),
    verbose_cout (std::cout, false),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process (mpi_communicator)
            == 0)),
    paper_data_out (paper_data_std, I_am_host),
    computing_timer (MPI_COMM_WORLD,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::cpu_and_wall_times),
    CFL_number (para_ptr_in->CFL_number),
    n_sparsity_pattern_out (-1),
    field_output_counter (0)
  {
    Assert (parameters, ExcMessage ("Null pointer encountered!"));

    continuation_coefficient = std::max (0.0, parameters->laplacian_continuation);

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

    if (I_am_host)
      {
        paper_data_std.open ("paper_data.txt");
        Assert (paper_data_std, ExcFileNotOpen ("paper_data.txt"));
        paper_data_std.setf (std::ios::scientific);
        paper_data_std.precision (6);
        unsigned int column = 0;

        paper_data_out << "#" << ++column << "  n_iter  ";
        paper_data_out << "#" << ++column << " n_step  ";
        paper_data_out << "#" << ++column << " C_conti..  ";
        paper_data_out << "#" << ++column << " C_time     ";
        paper_data_out << "#" << ++column << " C_lapla..  ";
        paper_data_out << "#" << ++column << " L2_res     ";
        paper_data_out << "#" << ++column << " L2_resPhy";
        paper_data_out << std::endl;
      }

    EulerEquations<dim>::set_parameter (parameters);

    verbose_cout.set_condition (parameters->output == Parameters::Solver::verbose);

    if (parameters->n_mms == 1)
      {
        // Setup coefficients for MMS
        std::array<Coeff, EquationComponents<dim>::n_components> coeffs;
        if (dim == 2)
          {
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
          }
        else if (dim == 3)
          {
            // component u:
            coeffs[0].c0 = 0.2;
            coeffs[0].cx = 0.001;
            coeffs[0].cy = -0.0035;
            coeffs[0].cz = 0.002;
            coeffs[0].ax = 1.5;
            coeffs[0].ay = 0.6;
            coeffs[0].az = 0.5;

            // component v:
            coeffs[1].c0 = 0.2;
            coeffs[1].cx = -0.001;
            coeffs[1].cy = 0.004;
            coeffs[1].cz = -0.0015;
            coeffs[1].ax = 0.5;
            coeffs[1].ay = 2.0/3.0;
            coeffs[1].az = 1.25;

            // component w:
            coeffs[2].c0 = 0.2;
            coeffs[2].cx = 0.0036;
            coeffs[2].cy = -0.002;
            coeffs[2].cz = -0.0011;
            coeffs[2].ax = 1.0/3.0;
            coeffs[2].ay = 1.5;
            coeffs[2].az = 1.0;

            // component density:
            coeffs[3].c0 = 1.0;
            coeffs[3].cx = 0.0015;
            coeffs[3].cy = -0.001;
            coeffs[3].cz = -0.0012;
            coeffs[3].ax = 1.0;
            coeffs[3].ay = 0.5;
            coeffs[3].az = 1.5;

            // component pressure:
            coeffs[4].c0 = 1.0/1.4;
            coeffs[4].cx = 0.0013;
            coeffs[4].cy = 0.005;
            coeffs[4].cz = -0.002;
            coeffs[4].ax = 2.0;
            coeffs[4].ay = 1.0;
            coeffs[4].az = 1.0/3.0;
          }
        // Initialize MMS
        mms.reinit (coeffs);
      }
  }

  template <int dim>
  NSolver<dim>::~NSolver()
  {
    paper_data_std.close();
    delete mapping_ptr;
  }

#include "NSolver.inst"
}

