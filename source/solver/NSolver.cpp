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

