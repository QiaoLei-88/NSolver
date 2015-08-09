//
//  prj01-Newton01.cpp
//  prj01-Newton2D
//
//  Created by Lei Qiao on 15/2/3.
//  A work based on deal.II turorial step-33.
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
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    mapping(),
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


  // @sect4{NSolver::assemble_face_term}
  //
  // Here, we do essentially the same as in the previous function. At the top,
  // we introduce the independent variables. Because the current function is
  // also used if we are working on an internal face between two cells, the
  // independent variables are not only the degrees of freedom on the current
  // cell but in the case of an interior face also the ones on the neighbor.
  template <int dim>
  void
  NSolver<dim>::assemble_face_term (const unsigned int           face_no,
                                    const FEFaceValuesBase<dim> &fe_v,
                                    const FEFaceValuesBase<dim> &fe_v_neighbor,
                                    const std::vector<types::global_dof_index>   &dof_indices,
                                    const std::vector<types::global_dof_index>   &dof_indices_neighbor,
                                    const bool                   external_face,
                                    const unsigned int           boundary_id,
                                    const double                 face_diameter)
  {
    const unsigned int n_q_points = fe_v.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

    std::vector<Sacado::Fad::DFad<double> >
    independent_local_dof_values (dofs_per_cell),
                                 independent_neighbor_dof_values (external_face == false ?
                                     dofs_per_cell :
                                     0);

    const unsigned int n_independent_variables = (external_face == false ?
                                                  2 * dofs_per_cell :
                                                  dofs_per_cell);

    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        independent_local_dof_values[i] = current_solution (dof_indices[i]);
        independent_local_dof_values[i].diff (i, n_independent_variables);
      }

    if (external_face == false)
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          independent_neighbor_dof_values[i]
            = current_solution (dof_indices_neighbor[i]);
          independent_neighbor_dof_values[i]
          .diff (i+dofs_per_cell, n_independent_variables);
        }


    // Next, we need to define the values of the conservative variables
    // ${\mathbf W}$ on this side of the face ($ {\mathbf W}^+$)
    // and on the opposite side (${\mathbf W}^-$), for both ${\mathbf W} =
    // {\mathbf W}^k_{n+1}$ and  ${\mathbf W} = {\mathbf W}_n$.
    // The "this side" values can be
    // computed in exactly the same way as in the previous function, but note
    // that the <code>fe_v</code> variable now is of type FEFaceValues or
    // FESubfaceValues:
    Table<2,Sacado::Fad::DFad<double> >
    Wplus (n_q_points, EquationComponents<dim>::n_components),
          Wminus (n_q_points, EquationComponents<dim>::n_components);
    Table<2,double>
    Wplus_old (n_q_points, EquationComponents<dim>::n_components),
              Wminus_old (n_q_points, EquationComponents<dim>::n_components);

    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const unsigned int component_i = fe_v.get_fe().system_to_component_index (i).first;
          Wplus[q][component_i] +=  independent_local_dof_values[i] *
                                    fe_v.shape_value_component (i, q, component_i);
          Wplus_old[q][component_i] +=  old_solution (dof_indices[i]) *
                                        fe_v.shape_value_component (i, q, component_i);
        }

    // Computing "opposite side" is a bit more complicated. If this is
    // an internal face, we can compute it as above by simply using the
    // independent variables from the neighbor:
    if (external_face == false)
      {
        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int component_i = fe_v_neighbor.get_fe().
                                               system_to_component_index (i).first;
              Wminus[q][component_i] += independent_neighbor_dof_values[i] *
                                        fe_v_neighbor.shape_value_component (i, q, component_i);
              Wminus_old[q][component_i] += old_solution (dof_indices_neighbor[i])*
                                            fe_v_neighbor.shape_value_component (i, q, component_i);
            }
      }
    // On the other hand, if this is an external boundary face, then the
    // values of $W^-$ will be either functions of $W^+$, or they will be
    // prescribed, depending on the kind of boundary condition imposed here.
    //
    // To start the evaluation, let us ensure that the boundary id specified
    // for this boundary is one for which we actually have data in the
    // parameters object. Next, we evaluate the function object for the
    // inhomogeneity.  This is a bit tricky: a given boundary might have both
    // prescribed and implicit values.  If a particular component is not
    // prescribed, the values evaluate to zero and are ignored below.
    //
    // The rest is done by a function that actually knows the specifics of
    // Euler equation boundary conditions. Note that since we are using fad
    // variables here, sensitivities will be updated appropriately, a process
    // that would otherwise be tremendously complicated.
    else
      {
        Assert (boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
                ExcIndexRange (boundary_id, 0,
                               Parameters::AllParameters<dim>::max_n_boundaries));

        std::vector<Vector<double> >
        boundary_values (n_q_points, Vector<double> (EquationComponents<dim>::n_components));

        if (parameters->n_mms != 1)
          {
            parameters->boundary_conditions[boundary_id]
            .values.vector_value_list (fe_v.get_quadrature_points(),
                                       boundary_values);
          }
        if (parameters->n_mms == 1)
          // MMS: compute boundary_values accroding to MS.
          {
            for (unsigned int q = 0; q < n_q_points; q++)
              {
                const Point<dim> p = fe_v.quadrature_point (q);
                std_cxx11::array<double, EquationComponents<dim>::n_components> sol, src;
                mms.evaluate (p,sol,src,false);
                for (unsigned int ic=0; ic < EquationComponents<dim>::n_components; ++ic)
                  {
                    boundary_values[q][ic] = sol[ic];
                  }
              }
          }

        for (unsigned int q = 0; q < n_q_points; q++)
          {
            EulerEquations<dim>::compute_Wminus (parameters->boundary_conditions[boundary_id].kind,
                                                 fe_v.normal_vector (q),
                                                 Wplus[q],
                                                 boundary_values[q],
                                                 Wminus[q]);
            // Here we assume that boundary type, boundary normal vector and boundary data values
            // maintain the same during time advancing.
            EulerEquations<dim>::compute_Wminus (parameters->boundary_conditions[boundary_id].kind,
                                                 fe_v.normal_vector (q),
                                                 Wplus_old[q],
                                                 boundary_values[q],
                                                 Wminus_old[q]);
          }
      }

    // Now that we have $\mathbf w^+$ and $\mathbf w^-$, we can go about
    // computing the numerical flux function $\mathbf H(\mathbf w^+,\mathbf
    // w^-, \mathbf n)$ for each quadrature point. Before calling the function
    // that does so, we also need to determine the Lax-Friedrich's stability
    // parameter:
    std::vector< std_cxx11::array < Sacado::Fad::DFad<double>, EquationComponents<dim>::n_components> >  normal_fluxes (
      n_q_points);
    std::vector< std_cxx11::array < double, EquationComponents<dim>::n_components> >  normal_fluxes_old (n_q_points);


    double alpha;

    switch (parameters->stabilization_kind)
      {
      case Parameters::Flux::constant:
        alpha = parameters->stabilization_value;
        break;
      case Parameters::Flux::mesh_dependent:
        alpha = face_diameter/ (2.0*time_step);
        break;
      default:
        Assert (false, ExcNotImplemented());
        alpha = 1;
      }

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        EulerEquations<dim>::numerical_normal_flux (fe_v.normal_vector (q),
                                                    Wplus[q], Wminus[q], alpha,
                                                    normal_fluxes[q],
                                                    parameters->numerical_flux_type);
        EulerEquations<dim>::numerical_normal_flux (fe_v.normal_vector (q),
                                                    Wplus_old[q], Wminus_old[q], alpha,
                                                    normal_fluxes_old[q],
                                                    parameters->numerical_flux_type);
      }

    // Now assemble the face term in exactly the same way as for the cell
    // contributions in the previous function. The only difference is that if
    // this is an internal face, we also have to take into account the
    // sensitivities of the residual contributions to the degrees of freedom on
    // the neighboring cell:
    std::vector<double> residual_derivatives (dofs_per_cell);
    for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
      if (fe_v.get_fe().has_support_on_face (i, face_no) == true)
        {
          Sacado::Fad::DFad<double> R_i = 0;

          for (unsigned int point=0; point<n_q_points; ++point)
            {
              const unsigned int
              component_i = fe_v.get_fe().system_to_component_index (i).first;

              R_i += (parameters->theta * normal_fluxes[point][component_i] +
                      (1.0 - parameters->theta) * normal_fluxes_old[point][component_i]) *
                     fe_v.shape_value_component (i, point, component_i) *
                     fe_v.JxW (point);
            }

          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              residual_derivatives[k] = R_i.fastAccessDx (k);
            }
          system_matrix.add (dof_indices[i], dof_indices, residual_derivatives);

          if (external_face == false)
            {
              for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                  residual_derivatives[k] = R_i.fastAccessDx (dofs_per_cell+k);
                }
              system_matrix.add (dof_indices[i], dof_indices_neighbor,
                                 residual_derivatives);
            }

          right_hand_side (dof_indices[i]) -= R_i.val();
        }
  }


  // @sect4{NSolver::solve}
  //
  // Here, we actually solve the linear system, using either of Trilinos'
  // Aztec or Amesos linear solvers. The result of the computation will be
  // written into the argument vector passed to this function. The result is a
  // pair of number of iterations and the final linear residual.

  template <int dim>
  std::pair<unsigned int, double>
  NSolver<dim>::solve (NSVector &newton_update)
  {
    switch (parameters->solver)
      {
      // If the parameter file specified that a direct solver shall be used,
      // then we'll get here. The process is straightforward, since deal.II
      // provides a wrapper class to the Amesos direct solver within
      // Trilinos. All we have to do is to create a solver control object
      // (which is just a dummy object here, since we won't perform any
      // iterations), and then create the direct solver object. When
      // actually doing the solve, note that we don't pass a
      // preconditioner. That wouldn't make much sense for a direct solver
      // anyway.  At the end we return the solver control statistics &mdash;
      // which will tell that no iterations have been performed and that the
      // final linear residual is zero, absent any better information that
      // may be provided here:
      case Parameters::Solver::direct:
      {
        SolverControl solver_control (1,0);
        TrilinosWrappers::SolverDirect direct (solver_control,
                                               parameters->output ==
                                               Parameters::Solver::verbose);

        direct.solve (system_matrix, newton_update, right_hand_side);

        return std::pair<unsigned int, double> (solver_control.last_step(),
                                                solver_control.last_value());
      }

      // Likewise, if we are to use an iterative solver, we use Aztec's GMRES
      // solver. We could use the Trilinos wrapper classes for iterative
      // solvers and preconditioners here as well, but we choose to use an
      // Aztec solver directly. For the given problem, Aztec's internal
      // preconditioner implementations are superior over the ones deal.II has
      // wrapper classes to, so we use ILU-T preconditioning within the
      // AztecOO solver and set a bunch of options that can be changed from
      // the parameter file.
      //
      // There are two more practicalities: Since we have built our right hand
      // side and solution vector as deal.II Vector objects (as opposed to the
      // matrix, which is a Trilinos object), we must hand the solvers
      // Trilinos Epetra vectors.  Luckily, they support the concept of a
      // 'view', so we just send in a pointer to our deal.II vectors. We have
      // to provide an Epetra_Map for the vector that sets the parallel
      // distribution, which is just a dummy object in serial. The easiest way
      // is to ask the matrix for its map, and we're going to be ready for
      // matrix-vector products with it.
      //
      // Secondly, the Aztec solver wants us to pass a Trilinos
      // Epetra_CrsMatrix in, not the deal.II wrapper class itself. So we
      // access to the actual Trilinos matrix in the Trilinos wrapper class by
      // the command trilinos_matrix(). Trilinos wants the matrix to be
      // non-constant, so we have to manually remove the constantness using a
      // const_cast.
      case Parameters::Solver::gmres:
      {
        Epetra_Vector x (View, system_matrix.trilinos_matrix().DomainMap(),
                         newton_update.begin());
        Epetra_Vector b (View, system_matrix.trilinos_matrix().RangeMap(),
                         right_hand_side.begin());

        AztecOO solver;
        solver.SetAztecOption (AZ_output,
                               (parameters->output ==
                                Parameters::Solver::quiet
                                ?
                                AZ_none
                                :
                                AZ_all));
        solver.SetAztecOption (AZ_solver, AZ_gmres);
        solver.SetRHS (&b);
        solver.SetLHS (&x);

        solver.SetAztecOption (AZ_precond,         AZ_dom_decomp);
        solver.SetAztecOption (AZ_subdomain_solve, AZ_ilut);
        solver.SetAztecOption (AZ_overlap,         0);
        solver.SetAztecOption (AZ_reorder,  parameters->AZ_RCM_reorder);

        solver.SetAztecParam (AZ_drop,      parameters->ilut_drop);
        solver.SetAztecParam (AZ_ilut_fill, parameters->ilut_fill);
        solver.SetAztecParam (AZ_athresh,   parameters->ilut_atol);
        solver.SetAztecParam (AZ_rthresh,   parameters->ilut_rtol);
        solver.SetUserMatrix (const_cast<Epetra_CrsMatrix *>
                              (&system_matrix.trilinos_matrix()));

        solver.Iterate (parameters->max_iterations, parameters->linear_residual);

        return std::pair<unsigned int, double> (solver.NumIters(),
                                                solver.TrueResidual());
      }
      }

    Assert (false, ExcNotImplemented());
    return std::pair<unsigned int, double> (0,0);
  }

  template <int dim>
  void
  NSolver<dim>::compute_refinement_indicators()
  {
    NSVector      tmp_vector;
    tmp_vector.reinit (current_solution, true);
    tmp_vector = predictor;

    switch (parameters->refinement_indicator)
      {
      case Parameters::Refinement<dim>::Gradient:
      {

        EulerEquations<dim>::compute_refinement_indicators (dof_handler,
                                                            mapping,
                                                            tmp_vector,
                                                            refinement_indicators,
                                                            parameters->component_mask);
        break;
      }
      case Parameters::Refinement<dim>::Kelly:
      {
        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1> (3),
                                            typename FunctionMap<dim>::type(),
                                            tmp_vector,
                                            refinement_indicators,
                                            parameters->component_mask);
        break;
      }
      default:
      {
        Assert (false, ExcNotImplemented());
        break;
      }
      }
  }
  // @sect4{NSolver::refine_grid}

  // Here, we use the refinement indicators computed before and refine the
  // mesh. At the beginning, we loop over all cells and mark those that we
  // think should be refined:
  template <int dim>
  void
  NSolver<dim>::refine_grid()
  {
    switch (parameters->refinement_indicator)
      {
      case Parameters::Refinement<dim>::Gradient:
      {
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
          if (cell->is_locally_owned())
            {
              cell->clear_coarsen_flag();
              cell->clear_refine_flag();

              if ((cell->level() < parameters->shock_levels) &&
                  (std::fabs (refinement_indicators (cell_no)) > parameters->shock_val))
                {
                  cell->set_refine_flag();
                }
              else if ((cell->level() > 0) &&
                       (std::fabs (refinement_indicators (cell_no)) < 0.75*parameters->shock_val))
                {
                  cell->set_coarsen_flag();
                }
            }
        break;
      }
      case Parameters::Refinement<dim>::Kelly:
      {
        parallel::distributed::GridRefinement::
        refine_and_coarsen_fixed_number (triangulation,
                                         refinement_indicators,
                                         parameters->refine_fraction,
                                         parameters->coarsen_fraction,
                                         parameters->max_cells);
        break;
      }
      default:
      {
        Assert (false, ExcNotImplemented());
        break;
      }
      }

    // Then we need to transfer the various solution vectors from the old to
    // the new grid while we do the refinement. The SolutionTransfer class is
    // our friend here; it has a fairly extensive documentation, including
    // examples, so we won't comment much on the following code. The last
    // three lines simply re-set the sizes of some other vectors to the now
    // correct size:

    NSVector tmp_vector;
    tmp_vector.reinit (old_solution, true);
    tmp_vector = predictor;
    // transfer_in needs vectors with ghost cells.
    std::vector<const NSVector * > transfer_in;
    transfer_in.push_back (&old_solution);
    transfer_in.push_back (&tmp_vector);

    parallel::distributed::SolutionTransfer<dim, NSVector> soltrans (dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    soltrans.prepare_for_coarsening_and_refinement (transfer_in);

    triangulation.execute_coarsening_and_refinement();

    setup_system();

    // Transfer data out
    {
      std::vector<NSVector * > transfer_out;
      NSVector interpolated_old_solution (predictor);
      NSVector interpolated_predictor (predictor);
      // transfer_out needs vectors without ghost cells.
      transfer_out.push_back (&interpolated_old_solution);
      transfer_out.push_back (&interpolated_predictor);
      soltrans.interpolate (transfer_out);
      old_solution = interpolated_old_solution;
      predictor = interpolated_predictor;
      current_solution = old_solution;
    }
  }


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
      const std::string data_name ("entropy_viscosity");
      data_out.add_data_vector (entropy_viscosity,
                                data_name,
                                DataOut<dim>::type_cell_data);
    }
    {
      const std::string data_name ("cellSize_viscosity");
      data_out.add_data_vector (cellSize_viscosity,
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
    for (unsigned int i=0; i<subdomain.size(); ++i)
      {
        subdomain (i) = triangulation.locally_owned_subdomain();
      }
    data_out.add_data_vector (subdomain, "subdomain");


    data_out.build_patches();

    static unsigned int output_file_number = 0;

    const std::string output_tag = "solution-" +
                                   Utilities::int_to_string (output_file_number, 4);

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

    ++output_file_number;
  }




  // @sect4{NSolver::run}

  // This function contains the top-level logic of this program:
  // initialization, the time loop, and the inner Newton iteration.
  //
  // At the beginning, we read the mesh file specified by the parameter file,
  // setup the DoFHandler and various vectors, and then interpolate the given
  // initial conditions on this mesh. We then perform a number of mesh
  // refinements, based on the initial conditions, to obtain a mesh that is
  // already well adapted to the starting solution. At the end of this
  // process, we output the initial solution.
  template <int dim>
  void NSolver<dim>::run()
  {
    computing_timer.enter_subsection ("0:Read grid");
    {
      if (parameters->n_mms == 1)
        {
          //MMS: build mesh directly.
          std::cerr << "  *** hyper_cube ***" << std::endl;
          GridGenerator::hyper_cube (triangulation, 0, 1);
          for (typename Triangulation<dim>::active_cell_iterator
               cell = triangulation.begin_active();
               cell != triangulation.end();
               ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face (f)->at_boundary())
                {
                  cell->face (f)->set_boundary_id (5);
                }

        }
      else
        {
          GridIn<dim> grid_in;
          grid_in.attach_triangulation (triangulation);

          std::ifstream input_file (parameters->mesh_filename.c_str());
          Assert (input_file, ExcFileNotOpen (parameters->mesh_filename.c_str()));

          if (parameters->mesh_format == Parameters::AllParameters<dim>::format_ucd)
            {
              grid_in.read_ucd (input_file);
            }
          if (parameters->mesh_format == Parameters::AllParameters<dim>::format_gmsh)
            {
              grid_in.read_msh (input_file);
            }

          if (parameters->scale_mesh != 1.0)
            {
              GridTools::scale (parameters->scale_mesh,triangulation);
            }
        }
      if (parameters->n_global_refinement > 0)
        {
          triangulation.refine_global (parameters->n_global_refinement);
        }
    }
    computing_timer.leave_subsection ("0:Read grid");
    computing_timer.enter_subsection ("1:Initialization");

    if (parameters->max_cells < 0.0)
      {
        parameters->max_cells = std::max (1.0, - (parameters->max_cells));
        parameters->max_cells *= triangulation.n_global_active_cells();
      }
    else
      {
        parameters->max_cells = std::max (parameters->max_cells,
                                          static_cast<double> (triangulation.n_global_active_cells()));
      }

    std::ofstream iteration_history_file_std;
    std::ofstream time_advance_history_file_std;
    if (I_am_host)
      {
        time_advance_history_file_std.open (
          parameters->time_advance_history_filename.c_str());
        Assert (time_advance_history_file_std,
                ExcFileNotOpen (parameters->time_advance_history_filename.c_str()));

        time_advance_history_file_std.setf (std::ios::scientific);
        time_advance_history_file_std.precision (6);

        iteration_history_file_std.open (
          parameters->iteration_history_filename.c_str());
        Assert (iteration_history_file_std,
                ExcFileNotOpen (parameters->iteration_history_filename.c_str()));
        iteration_history_file_std.setf (std::ios::scientific);
        iteration_history_file_std.precision (6);
      }
    ConditionalOStream iteration_history_file (iteration_history_file_std,
                                               I_am_host);
    ConditionalOStream time_advance_history_file (time_advance_history_file_std,
                                                  I_am_host);


    setup_system();

    initialize();
    if (parameters->do_refine)
      {
        refine_grid();
        initialize();
      }

    check_negative_density_pressure();
    calc_time_step();
    output_results();

    // We then enter into the main time stepping loop. At the top we simply
    // output some status information so one can keep track of where a
    // computation is, as well as the header for a table that indicates
    // progress of the nonlinear inner iteration:

    double time = 0;
    double next_output = time + parameters->output_step;
    double old_time_step_size = time_step;

    predictor = old_solution;

    bool newton_iter_converged (false);
    bool CFL_number_increased (false);
    int index_linear_search_length (0);
    const double linear_search_length[12] = {1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 1.2, 1.5, 2.0};
    unsigned int converged_newton_iters (0);

    unsigned int n_time_step (0);
    unsigned int n_total_inter (0);

    time_advance_history_file
        << "   iter     n_cell     n_dofs          time   i_step"
        << "  i_Newton    Newton_res  n_linear_iter    linear_res"
        << "  linear_search_len  time_step_size    CFL_number"
        << "  time_march_res"
        << '\n';
    iteration_history_file
        << "   iter     n_cell     n_dofs          time   i_step"
        << "  i_Newton    Newton_res  n_linear_iter    linear_res"
        << "  linear_search_len  time_step_size    CFL_number"
        << "  Newton_update_norm"
        << '\n';

    NSVector      tmp_vector;

    computing_timer.leave_subsection ("1:Initialization");
    bool terminate_time_stepping (false);
    while (!terminate_time_stepping)
      {
        computing_timer.enter_subsection ("2:Prepare Newton iteration");
        if (parameters->is_steady)
          {
            pcout << "Step = " << n_time_step << std::endl;
          }
        else
          {
            pcout << "T = " << time << std::endl;
          }
        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << std::endl;

        pcout << "   NonLin Res   NewtonUpdateNorm  Lin Iter     Lin Res     "
              << "Linear Search Len      Time Step Size      CFL number" << std::endl
              << "   __________________________________________"
              << "_____________________________________________" << std::endl;

        // Then comes the inner Newton iteration to solve the nonlinear
        // problem in each time step. The way it works is to reset matrix and
        // right hand side to zero, then assemble the linear system. If the
        // norm of the right hand side is small enough, then we declare that
        // the Newton iteration has converged. Otherwise, we solve the linear
        // system, update the current solution with the Newton increment, and
        // output convergence information. At the end, we check that the
        // number of Newton iterations is not beyond a limit of 10 -- if it
        // is, it appears likely that iterations are diverging and further
        // iterations would do no good. If that happens, we throw an exception
        // that will be caught in <code>main()</code> with status information
        // being displayed before the program aborts.
        //
        // Note that the way we write the AssertThrow macro below is by and
        // large equivalent to writing something like <code>if (!(nonlin_iter
        // @<= 10)) throw ExcMessage ("No convergence in nonlinear
        // solver");</code>. The only significant difference is that
        // AssertThrow also makes sure that the exception being thrown carries
        // with it information about the location (file name and line number)
        // where it was generated. This is not overly critical here, because
        // there is only a single place where this sort of exception can
        // happen; however, it is generally a very useful tool when one wants
        // to find out where an error occurred.


        newton_iter_converged = false;

        unsigned int nonlin_iter = 0;
        current_solution = predictor;
        bool linear_solver_diverged (true);
        unsigned int const nonlin_iter_threshold (10);
        double const nonlin_iter_tolerance (
          parameters->is_steady ? 1.0e+20 : 1.0e-10);
        double reference_nonlin_residual (1.0);
        double nonlin_residual_ratio (1.0);


        double res_norm;
        double newton_update_norm;
        std::pair<unsigned int, double> convergence;

        locally_owned_solution = current_solution;

        computing_timer.leave_subsection ("2:Prepare Newton iteration");
        do // Newton iteration
          {
            computing_timer.enter_subsection ("3:Assemble Newton system");
            system_matrix = 0;
            right_hand_side = 0;

            assemble_system (nonlin_iter);

            res_norm = right_hand_side.l2_norm();
            if (nonlin_iter == 0)
              {
                residual_for_output = right_hand_side;
              }

            newton_update = 0;

            computing_timer.leave_subsection ("3:Assemble Newton system");

            computing_timer.enter_subsection ("4:Solve Newton system");
            convergence = solve (newton_update);
            computing_timer.leave_subsection ("4:Solve Newton system");

            computing_timer.enter_subsection ("5:Postprocess Newton solution");
            Assert (index_linear_search_length < 9, ExcIndexRange (index_linear_search_length,0,9));
            newton_update *= linear_search_length[index_linear_search_length];
            locally_owned_solution += newton_update;
            current_solution = locally_owned_solution;
            newton_update_norm = newton_update.l2_norm();
            if (I_am_host)
              {
                std::printf ("   %-13.6e    %-13.6e  %04d        %-5.2e            %7.4g          %7.4g          %7.4g\n",
                             res_norm,newton_update_norm, convergence.first, convergence.second,
                             linear_search_length[index_linear_search_length],
                             time_step, CFL_number);
              }
            linear_solver_diverged = std::isnan (convergence.second);

            ++nonlin_iter;
            ++n_total_inter;

            if (n_total_inter <= parameters-> n_iter_stage1)
              {
                reference_nonlin_residual = newton_update_norm;
              }
            else
              {
                nonlin_residual_ratio = reference_nonlin_residual/newton_update_norm;
              }


            // Out put convergence history
            iteration_history_file
                << std::setw (7) << n_total_inter << ' '
                << std::setw (10) << triangulation.n_global_active_cells() << ' '
                << std::setw (10) << dof_handler.n_dofs() << ' '
                << std::setw (13) << time << ' '
                << std::setw (8) << n_time_step << ' '
                << std::setw (9) << nonlin_iter << ' '
                << std::setw (13) << res_norm << ' '
                << std::setw (14) << convergence.first << ' '
                << std::setw (13) << convergence.second << ' '
                << std::setw (18) << linear_search_length[index_linear_search_length] << ' '
                << std::setw (15) << time_step << ' '
                << std::setw (13) << CFL_number << ' '
                << std::setw (19) << newton_update_norm
                << '\n';
            // Check result.
            if (res_norm < nonlin_iter_tolerance)
              {
                newton_iter_converged = true;
              }

            if (linear_solver_diverged)
              {
                pcout << "  Linear solver diverged..\n";
              }
            // May 'newton_iter_converged' and 'linear_solver_diverged' be true
            // together? I don't think so but not sure.


            // If linear solver diverged or Newton interation not converge in
            // a reasonable interations, we terminate this time step and set
            // 'newton_iter_converged' to false. Then further action will be taken
            // to handle this situation.

            //                                 Using '>='  here because this condition
            //                                 is evaluated after '++nonlin_iter'.
            if (linear_solver_diverged || nonlin_iter >= nonlin_iter_threshold)
              {
                newton_iter_converged = false;
                pcout << "  Newton iteration not converge in " << nonlin_iter_threshold << " steps.\n";
              }
            computing_timer.leave_subsection ("5:Postprocess Newton solution");
          }
        while ((!newton_iter_converged)
               && nonlin_iter < nonlin_iter_threshold
               && (!linear_solver_diverged));

        if (newton_iter_converged)
          {
            computing_timer.enter_subsection ("6:Postprocess time step");

            WallForce wall_force;
            integrate_force (parameters, wall_force);


            pcout << "  Lift and drag:" << std::endl
                  << "  " << wall_force.lift
                  << "  " << wall_force.drag << std::endl
                  << std::endl;

            pcout << "  Force_x,y,z:" << std::endl
                  << "  ";
            for (unsigned int id=0; id<3; ++id)
              {
                pcout << "  " << wall_force.force[id];
              }
            pcout << std::endl
                  << "  Moment,x,y,z:" << std::endl
                  << "  ";
            for (unsigned int id=0; id<3; ++id)
              {
                pcout << "  " << wall_force.moment[id];
              }
            pcout << std::endl;


            //Output time marching history
            time_advance_history_file
                << std::setw (7) << n_total_inter << ' '
                << std::setw (10) << triangulation.n_global_active_cells() << ' '
                << std::setw (10) << dof_handler.n_dofs() << ' '
                << std::setw (13) << time << ' '
                << std::setw (8) << n_time_step << ' '
                << std::setw (9) << nonlin_iter << ' '
                << std::setw (13) << res_norm << ' '
                << std::setw (14) << convergence.first << ' '
                << std::setw (13) << convergence.second << ' '
                << std::setw (18) << linear_search_length[index_linear_search_length] << ' '
                << std::setw (15) << time_step << ' '
                << std::setw (13) << CFL_number << ' ';

            // We only get to this point if the Newton iteration has converged, so
            // do various post convergence tasks here:
            //
            // First, we update the time and produce graphical output if so
            // desired. Then we update a predictor for the solution at the next
            // time step by approximating $\mathbf w^{n+1}\approx \mathbf w^n +
            // \delta t \frac{\partial \mathbf w}{\partial t} \approx \mathbf w^n
            // + \delta t \; \frac{\mathbf w^n-\mathbf w^{n-1}}{\delta t} = 2
            // \mathbf w^n - \mathbf w^{n-1}$ to try and make adaptivity work
            // better.  The idea is to try and refine ahead of a front, rather
            // than stepping into a coarse set of elements and smearing the
            // old_solution.  This simple time extrapolator does the job. With
            // this, we then refine the mesh if so desired by the user, and
            // finally continue on with the next time step:
            ++converged_newton_iters;
            if (!parameters->is_steady)
              {
                time += time_step;
              }
            ++n_time_step;
            terminate_time_stepping = terminate_time_stepping || time >= parameters->final_time;
            terminate_time_stepping = terminate_time_stepping ||
                                      (parameters->is_steady &&
                                       n_total_inter >= parameters -> max_Newton_iter);

            if (parameters->do_refine == true)
              {
                compute_refinement_indicators();
              }

            if (parameters->output_step < 0)
              {
                output_results();
              }
            else if (time >= next_output)
              {
                output_results();
                next_output += parameters->output_step;
              }

            if (parameters-> is_steady)
              {
                if (n_total_inter <= parameters-> n_iter_stage1)
                  {
                    CFL_number *= parameters->step_increasing_ratio_stage1;
                  }
                else
                  {
                    double const
                    ratio = std::max (parameters->minimum_step_increasing_ratio_stage2,
                                      std::pow (nonlin_residual_ratio, parameters->step_increasing_power_stage2));
                    CFL_number *= ratio;
                  }
              }
            else if ((converged_newton_iters%10 == 0) &&
                     parameters->allow_increase_CFL &&
                     (CFL_number < parameters->CFL_number_max)
                    )
              {
                //Since every thing goes so well, let's try a larger time step next.
                CFL_number *= 2.0;
                CFL_number = std::min (CFL_number, parameters->CFL_number_max);
                CFL_number_increased = true;
                index_linear_search_length = 0;
              }

            std_cxx11::array<double, EquationComponents<dim>::n_components> time_advance_l2_norm;
            for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
              {
                mms_error_l2[ic] = 0.0;
                time_advance_l2_norm[ic] = 0.0;
              }
            mms_error_linfty = 0.0;

            std::vector<Vector<double> > solution_values;
            std::vector<Vector<double> > old_solution_values;
            //MMS: update quadrature points for evaluation of manufactored solution.
            const UpdateFlags update_flags = update_values
                                             | update_JxW_values
                                             | update_quadrature_points;
            const QGauss<dim> quadrature_error (parameters->error_quadrature_degree);
            FEValues<dim>  fe_v (mapping, fe, quadrature_error, update_flags);

            // Then integrate error over all cells,
            typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
            for (; cell!=endc; ++cell)
              if (cell->is_locally_owned())
                {
                  fe_v.reinit (cell);
                  const unsigned int n_q_points = fe_v.n_quadrature_points;

                  solution_values.resize (n_q_points,
                                          Vector<double> (EquationComponents<dim>::n_components));
                  old_solution_values.resize (n_q_points,
                                              Vector<double> (EquationComponents<dim>::n_components));
                  fe_v.get_function_values (current_solution, solution_values);
                  fe_v.get_function_values (old_solution, old_solution_values);

                  std::vector < std_cxx11::array< double, EquationComponents<dim>::n_components> >
                  mms_source (n_q_points), mms_value (n_q_points);

                  for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
                    {
                      for (unsigned int q=0; q<n_q_points; ++q)
                        {
                          time_advance_l2_norm[ic] += (old_solution_values[q][ic] - solution_values[q][ic]) *
                                                      (old_solution_values[q][ic] - solution_values[q][ic]) *
                                                      fe_v.JxW (q);
                        }
                    }
                  if (parameters->n_mms == 1)
                    {

                      for (unsigned int q=0; q<n_q_points; ++q)
                        {
                          mms.evaluate (fe_v.quadrature_point (q), mms_value[q], mms_source[q], /* const bool need_source = */ false);
                        }

                      for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
                        {
                          for (unsigned int q=0; q<n_q_points; ++q)
                            {
                              mms_error_l2[ic] += (mms_value[q][ic] - solution_values[q][ic]) *
                                                  (mms_value[q][ic] - solution_values[q][ic]) *
                                                  fe_v.JxW (q);
                            }
                        }
                    }
                }
            if (parameters->n_mms == 1)
              {
                pcout << "  Error Info:\n";
                pcout << "    n_dofs    u_err    v_err  rho_err    p_err (log10)\n   ";
                pcout <<  std::log10 (dof_handler.n_dofs()) << ' ';
                for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
                  {
                    Utilities::MPI::sum (mms_error_l2[ic], mpi_communicator);
                    pcout << 0.5 * std::log10 (mms_error_l2[ic]) << ' ';
                  }
                pcout << std::endl;
              }

            // Only try to switch flux type when current and target flux
            // types are different.
            bool swith_flux =
              parameters->numerical_flux_type != parameters->flux_type_switch_to;
            // If switch of flux type is requested and the flux switch tolerance is
            // larger than time march tolerance, never stop time marching
            // beforce switching to the target flux type.
            bool time_march_converged =
              ! (swith_flux &&
                 parameters->tolerance_to_switch_flux > parameters->time_march_tolerance);

            pcout << "  Order of time advancing L_2  norm\n   ";
            double total_time_march_norm = 0.0;
            for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
              {
                Utilities::MPI::sum (time_advance_l2_norm[ic], mpi_communicator);
                total_time_march_norm += time_advance_l2_norm[ic];
                double const log_norm = 0.5 * std::log10 (time_advance_l2_norm[ic]);
                pcout << log_norm << ' ';
                time_march_converged = time_march_converged &&
                                       (log_norm < parameters->time_march_tolerance);
                swith_flux = swith_flux &&
                             (log_norm < parameters->tolerance_to_switch_flux);
              }
            time_advance_history_file << std::setw (15)
                                      << std::sqrt (total_time_march_norm) << '\n';
            pcout << std::endl;

            if (swith_flux)
              {
                parameters->numerical_flux_type = parameters->flux_type_switch_to;
              }
            terminate_time_stepping = terminate_time_stepping ||
                                      (parameters->is_steady &&
                                       time_march_converged);

            old_old_solution = old_solution;
            old_solution = current_solution;
            if (parameters->do_refine == true)
              {
                refine_grid();
              }

            old_time_step_size = time_step;
            check_negative_density_pressure();
            calc_time_step();
            // Uncomment the following line if you want reset the linear_search_length immediatly after a converged Newton iter.
            //index_linear_search_length = 0;
            computing_timer.leave_subsection ("6:Postprocess time step");
          }
        else
          {
            computing_timer.enter_subsection ("6:Rolling back time step");
            // Newton iteration not converge in reasonable steps

            if ((index_linear_search_length <
                 parameters->newton_linear_search_length_try_limit)
                && (!CFL_number_increased))
              {
                // Try to adjust linear_search_length first
                ++index_linear_search_length;
              }
            else
              {
                AssertThrow (parameters->allow_decrease_CFL,
                             ExcMessage ("Nonlinear not convergence and reduceing CFL number is disabled."));
                // Reduce time step when linear_search_length has tried out.
                CFL_number *= 0.5;
                AssertThrow (CFL_number >= parameters->CFL_number_min,
                             ExcMessage ("No convergence in nonlinear solver after all small time step and linear search length trid out."));

                pcout << "  Recompute with different linear search length or time step...\n\n";
                time_step *= 0.5;
                CFL_number_increased = false;
                index_linear_search_length = 0;
              }
            // Reset counter
            converged_newton_iters = 0;
            computing_timer.leave_subsection ("6:Rolling back time step");
          }

        // Predict solution of next time step
        predictor = old_solution;
        if (! (parameters->is_steady))
          {
            // Only extrapolate predictor in unsteady simulation
            double const predict_ratio = parameters->solution_extrapolation_length *
                                         time_step / old_time_step_size;
            tmp_vector.reinit (predictor);
            tmp_vector  = old_old_solution;
            predictor.sadd (1.0+predict_ratio,
                            0.0-predict_ratio, tmp_vector);
          }

        time_advance_history_file << std::flush;
        iteration_history_file << std::flush;
      } // End of time advancing
    if (I_am_host)
      {
        time_advance_history_file_std.close();
        iteration_history_file_std.close();
      }
    // Timer initialized with TimerOutput::summary will print summery information
    // on its destruction.
    // computing_timer.print_summary();
  } //End of NSolver<dim>::run ()

#include "NSolver.inst"
}

