//
//  NSolver::run.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


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

      if (parameters->manifold_circle > 0)
        {
          // Hard coded test case: ManifoldCircle
          Assert (parameters->n_mms != 1,
                  ExcMessage ("MMS and C1Circle case can't play together!!!"));
          Assert (parameters->NACA_foil == 0,
                  ExcMessage ("NACA_foil and C1Circle case can't play together!!!"));

          for (typename Triangulation<dim>::active_cell_iterator
               cell = triangulation.begin_active();
               cell != triangulation.end();
               ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face (f)->at_boundary())
                {
                  if (cell->face (f)->boundary_id() == 2)
                    {
                      cell->face (f)->set_manifold_id (2);
                    }
                }
          if (parameters->manifold_circle == 1)
            {
              // Circle
              spherical_boundary = new HyperBallBoundary<dim> (Point<dim>()/*=(0,0,...)*/,/*radius=*/0.5);
            }
          else if (parameters->manifold_circle == 2)
            {
              //GAMM Channel
              spherical_boundary = new HyperBallBoundary<dim> (Point<dim> (0.0, -1.2), /*radius=*/1.3);
            }
          triangulation.set_boundary (2, *spherical_boundary);

          if (parameters->manifold_circle == 2
              &&
              // Borrow a parameter. This is not good. And I know its not good. Only this one time.
              parameters->NACA_cheating_refinement == 1)
            {
              for (int n=0; n<2; ++n)
                {
                  for (typename Triangulation<dim>::active_cell_iterator
                       cell = triangulation.begin_active();
                       cell != triangulation.end();
                       ++cell)
                    {
                      // Refine a elliptic region twice
                      const double center_x=0.0;
                      const double center_y=0.0;
                      const double half_axi_long  = 0.75;
                      const double half_axi_short = 0.5;
                      const double a = (cell->center()[0] - center_x)/half_axi_long;
                      const double b = (cell->center()[1] - center_y)/half_axi_short;
                      if ((a*a+b*b) < 1.0)
                        {
                          cell->set_refine_flag();
                        }
                      // And the downstream floor region
                      if (cell->center()[0] > 0.5 && cell->center()[1] < 0.2)
                        {
                          cell->set_refine_flag();
                        }
                    }
                  triangulation.execute_coarsening_and_refinement();
                }
              // refine the two stagnation points once more
              for (typename Triangulation<dim>::active_cell_iterator
                   cell = triangulation.begin_active();
                   cell != triangulation.end();
                   ++cell)
                {
                  Point<dim> front;
                  front[0] = -0.5;
                  Point<dim> rear;
                  rear[0] = 0.5;
                  if (front.distance (cell->center()) < 0.2
                      ||
                      rear.distance (cell->center()) < 0.2)
                    {
                      cell->set_refine_flag();
                    }
                }
              triangulation.execute_coarsening_and_refinement();
            } // End if (parameters->manifold_circle == 2 /* && cheating refinement == true*/)
        }
      if (parameters->NACA_foil > 0)
        {
          // Hard coded test case: NACA 4 digit foils
          Assert (parameters->n_mms != 1,
                  ExcMessage ("MMS and NACA_foil case can't play together!!!"));
          Assert (parameters->manifold_circle == 0,
                  ExcMessage ("NACA_foil and C1Circle case can't play together!!!"));
          for (typename Triangulation<dim>::active_cell_iterator
               cell = triangulation.begin_active();
               cell != triangulation.end();
               ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face (f)->at_boundary())
                {
                  if (cell->face (f)->boundary_id() == 1)
                    {
                      cell->face (f)->set_manifold_id (1);
                    }
                }
          triangulation.set_boundary (1, NACA_foil_boundary);

          if (parameters->NACA_cheating_refinement == 1)
            {
              for (int n=0; n<2; ++n)
                {
                  // Refine cell in a elliptic around the foil twice
                  for (typename Triangulation<dim>::active_cell_iterator
                       cell = triangulation.begin_active();
                       cell != triangulation.end();
                       ++cell)
                    {
                      const double center_x=0.5;
                      const double center_y=0.0;
                      const double half_axi_long  = 0.7;
                      const double half_axi_short = 0.2;
                      const double a = (cell->center()[0] - center_x)/half_axi_long;
                      const double b = (cell->center()[1] - center_y)/half_axi_short;
                      if ((a*a+b*b) < 1.0)
                        {
                          cell->set_refine_flag();
                        }
                    }
                  triangulation.execute_coarsening_and_refinement();
                }

              for (typename Triangulation<dim>::active_cell_iterator
                   cell = triangulation.begin_active();
                   cell != triangulation.end();
                   ++cell)
                {
                  const Point<dim> zero;
                  Point<dim> one;
                  one[0] = 1.0;
                  if (zero.distance (cell->center()) < 0.15
                      ||
                      one.distance (cell->center()) < 0.1)
                    {
                      cell->set_refine_flag();
                    }
                }
              triangulation.execute_coarsening_and_refinement();

              for (typename Triangulation<dim>::active_cell_iterator
                   cell = triangulation.begin_active();
                   cell != triangulation.end();
                   ++cell)
                {
                  const Point<dim> zero;
                  Point<dim> one;
                  one[0] = 1.0;
                  if (zero.distance (cell->center()) < 0.05
                      ||
                      one.distance (cell->center()) < 0.03)
                    {
                      cell->set_refine_flag();
                    }
                }
              triangulation.execute_coarsening_and_refinement();
            }
          if (parameters->NACA_cheating_refinement == 2)
            {
              for (int n=0; n<1; ++n)
                {
                  // Refine cell in a elliptic around the foil twice
                  for (typename Triangulation<dim>::active_cell_iterator
                       cell = triangulation.begin_active();
                       cell != triangulation.end();
                       ++cell)
                    {
                      const double center_x=0.5;
                      const double center_y=0.0;
                      const double half_axi_long  = 0.7;
                      const double half_axi_short = 0.2;
                      const double a = (cell->center()[0] - center_x)/half_axi_long;
                      const double b = (cell->center()[1] - center_y)/half_axi_short;
                      if ((a*a+b*b) < 1.0)
                        {
                          cell->set_refine_flag();
                        }
                    }
                  triangulation.execute_coarsening_and_refinement();
                }

              for (typename Triangulation<dim>::active_cell_iterator
                   cell = triangulation.begin_active();
                   cell != triangulation.end();
                   ++cell)
                {
                  // Refine LE and TE zone once more
                  const Point<dim> zero;
                  Point<dim> one;
                  one[0] = 1.0;
                  if (zero.distance (cell->center()) < 0.15
                      ||
                      one.distance (cell->center()) < 0.1)
                    {
                      cell->set_refine_flag();
                    }
                  const Point<dim> &p = cell->center();
                  // refine upper surface shock zone
                  {
                    bool refine = true;
                    refine = refine && (p[0] > 0.5);
                    refine = refine && (p[0] < 0.7);
                    refine = refine && (p[1] > 0.0);
                    refine = refine && (p[1] < 1.0);
                    if (refine)
                      {
                        cell->set_refine_flag();
                      }
                  }
                  // refine lower surface shock zone
                  {
                    bool refine = true;
                    refine = refine && (p[0] > 0.3);
                    refine = refine && (p[0] < 0.40);
                    refine = refine && (p[1] > -0.25);
                    refine = refine && (p[1] < 0.0);
                    if (refine)
                      {
                        cell->set_refine_flag();
                      }
                  }
                }
              triangulation.execute_coarsening_and_refinement();

              for (typename Triangulation<dim>::active_cell_iterator
                   cell = triangulation.begin_active();
                   cell != triangulation.end();
                   ++cell)
                {
                  const Point<dim> zero;
                  Point<dim> one;
                  one[0] = 1.0;
                  // Refine LE and TE zone once more in a smaller region
                  if (zero.distance (cell->center()) < 0.05
                      ||
                      one.distance (cell->center()) < 0.03)
                    {
                      cell->set_refine_flag();
                    }
                  // refine upper surface shock zone
                  const Point<dim> &p = cell->center();
                  {
                    bool refine = true;
                    refine = refine && (p[0] > 0.58);
                    refine = refine && (p[0] < 0.68);
                    refine = refine && (p[1] > 0.0);
                    refine = refine && (p[1] < 0.8);
                    if (refine)
                      {
                        cell->set_refine_flag();
                      }
                  }
                  // refine lower surface shock zone
                  {
                    bool refine = true;
                    refine = refine && (p[0] > 0.32);
                    refine = refine && (p[0] < 0.38);
                    refine = refine && (p[1] > -0.2);
                    refine = refine && (p[1] < 0.0);
                    if (refine)
                      {
                        cell->set_refine_flag();
                      }
                  }
                }
              triangulation.execute_coarsening_and_refinement();
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
        parameters_modifier->max_cells = std::max (1.0, - (parameters->max_cells));
        parameters_modifier->max_cells *= triangulation.n_global_active_cells();
      }
    else
      {
        parameters_modifier->max_cells = std::max (parameters->max_cells,
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
    if ((parameters->max_refine_time > 0) &&
        parameters->do_refine_on_initial_field)
      {
        compute_refinement_indicators();
        refine_grid();
        initialize();
      }

    check_negative_density_pressure();
    calc_time_step();
    calc_laplacian_indicator();
    output_results();

// We then enter into the main time stepping loop. At the top we simply
// output some status information so one can keep track of where a
// computation is, as well as the header for a table that indicates
// progress of the nonlinear inner iteration:

    double time = 0;
    double next_output = time + parameters->output_step;
    double old_time_step_size = global_time_step_size;

    predictor = old_solution;

    bool newton_iter_converged (false);
    bool CFL_number_increased (false);
    int index_linear_search_length (0);
    const double linear_search_length[12] = {1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 1.2, 1.5, 2.0};
    unsigned int converged_newton_iters (0);

    n_time_step = 0;
    const unsigned int int_output_step = static_cast<unsigned int> (parameters->output_step);
    unsigned int next_output_time_step (n_time_step + int_output_step);
    n_total_iter = 0;

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
    double res_norm_total (0.0);
    double res_norm_total_previous (0.0);
    double res_norm_infty_total (0.0);
    double old_continuation_coefficient = continuation_coefficient;
    unsigned int n_step_laplacian_vanished = 65535;
    bool is_refine_step = false;
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
              << "Linear Search Len      Time Step Size      CFL number   ||Res||_infty ||Res||_infty_all" << std::endl
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
        double terminal_res = 0.0;

        nonlin_iter = 0;
        current_solution = predictor;
        bool linear_solver_diverged (true);
        bool solution_diverged (true);
        unsigned int const nonlin_iter_threshold (parameters->max_Newton_iter);

        double res_norm;
        double res_norm_infty;
        double newton_update_norm;
        std::pair<unsigned int, double> convergence;

        locally_owned_solution = current_solution;

        computing_timer.leave_subsection ("2:Prepare Newton iteration");

        res_norm_total_previous = res_norm_total;
        res_norm_total = 0.0;
        res_norm_infty_total = 0.0;
        double physical_residual_first = 0.0;
        double physical_residual_ratio = 0.0;
        double log_res_last = 0.0;
        bool quadratic_converge = true;
        calc_artificial_viscosity();
        calc_laplacian_indicator();
        if (! (parameters->dodge_mesh_adaptation) || is_refine_step)
          {
            pcout << " *** refine step ***" << std::endl;
          }
        if (! (parameters->dodge_mesh_adaptation) || !is_refine_step)
          {
            pcout << " *** laplacian step ***" << std::endl;
          }
        // Set up continuation coefficients for pseudo time component and laplacian component
        switch (parameters->continuation_type)
          {
          case Parameters::StabilizationParameters<dim>::CT_timeCFL:
          {
            // Works are done by calc_time_step().
            // Nothing needs to do here.
            break;
          }
          case Parameters::StabilizationParameters<dim>::CT_time:
          {
            continuation_coeff_time = continuation_coefficient;
            continuation_coeff_laplacian = 0.0;
            break;
          }
          case Parameters::StabilizationParameters<dim>::CT_laplacian:
          {
            continuation_coeff_time = 0.0;
            continuation_coeff_laplacian = continuation_coefficient;
            break;
          }
          case Parameters::StabilizationParameters<dim>::CT_switch:
          {
            if (continuation_coefficient > mean_artificial_viscosity * parameters->continuation_switch_threshold)
              {
                continuation_coeff_time = 0.0;
                continuation_coeff_laplacian = continuation_coefficient;
              }
            else
              {
                continuation_coeff_time = continuation_coefficient;
                continuation_coeff_laplacian = 0.0;
              }
            break;
          }
          case Parameters::StabilizationParameters<dim>::CT_alternative:
          {
            if (n_time_step%2 == 0)
              {
                continuation_coeff_time = 0.0;
                continuation_coeff_laplacian = continuation_coefficient;
              }
            else
              {
                continuation_coeff_time = continuation_coefficient;
                continuation_coeff_laplacian = 0.0;
              }
            break;
          }
          case Parameters::StabilizationParameters<dim>::CT_blend:
          {
            // TODO: risk of dividing zero.
            const double weight
              = (0.5 * continuation_coefficient) / ((0.5 * continuation_coefficient) + mean_artificial_viscosity);
            continuation_coeff_time      = (1.0-weight) * continuation_coefficient;
            continuation_coeff_laplacian = weight       * continuation_coefficient;
            break;
          }
          default:
          {
            Assert (false, ExcNotImplemented());
            break;
          }
          }
        do // Newton iteration
          {
            computing_timer.enter_subsection ("3:Assemble Newton system");
            system_matrix = 0;
            right_hand_side = 0;
            physical_residual = 0;
            newton_update = 0;

            assemble_system();
            apply_strong_boundary_condtions();

            if (parameters->output_system_matrix)
              {
                const std::string file_name = "system_matrix.step_"
                                              + Utilities::int_to_string (n_time_step,4)
                                              + ".iter_"
                                              + Utilities::int_to_string (nonlin_iter,4)
                                              + ".MTX";
                std::ofstream out (file_name.c_str());
                Tools::write_matrix_MTX (out, system_matrix);
                out.close();
              }

            res_norm = right_hand_side.l2_norm();
            solution_diverged = std::isnan (res_norm);
            res_norm_total += res_norm;
            const double physical_res_norm = physical_residual.l2_norm();

            res_norm_infty = right_hand_side.linfty_norm();
            res_norm_infty_total += res_norm_infty;
            if (nonlin_iter == 0)
              {
                log_res_last = std::log (res_norm);
                physical_residual_first = physical_res_norm;
                residual_for_output = right_hand_side;
                terminal_res =
                  res_norm * std::pow (10.0, parameters->nonlinear_tolerance);
                if (parameters->laplacian_continuation > 0.0)
                  {
                    if (continuation_coefficient > 0.0)
                      {
                        terminal_res =
                          std::max (res_norm * 0.1,
                                    physical_res_norm * parameters->laplacian_newton_tolerance);
                      }
                    else
                      {
                        terminal_res = 1e-11;
                      }
                  }
              }
            else
              {
                const double log_res = std::log (res_norm);
                // This only works when log_res<0 && log_res_last<0.
                quadratic_converge =
                  (log_res < log_res_last * parameters->laplacian_newton_quadratic);
                log_res_last = log_res;
              }
            physical_residual_ratio = physical_res_norm/physical_residual_first;

            computing_timer.leave_subsection ("3:Assemble Newton system");

            computing_timer.enter_subsection ("4:Solve Newton system");
            if (!solution_diverged)
              {
                convergence = solve (newton_update);
              }
            computing_timer.leave_subsection ("4:Solve Newton system");

            computing_timer.enter_subsection ("5:Postprocess Newton solution");
            Assert (index_linear_search_length < 9, ExcIndexRange (index_linear_search_length,0,9));
            newton_update *= linear_search_length[index_linear_search_length];
            locally_owned_solution += newton_update;
            current_solution = locally_owned_solution;
            newton_update_norm = newton_update.l2_norm();
            if (I_am_host)
              {
                std::printf ("   %-13.6e    %-13.6e  %04d        %-5.2e            %7.4g          %7.4g          %7.4g      %11.4e    %11.4e    %11.4e\n",
                             res_norm,newton_update_norm, convergence.first, convergence.second,
                             linear_search_length[index_linear_search_length],
                             global_time_step_size, CFL_number, res_norm_infty, res_norm_infty_total,
                             physical_res_norm);
              }
            linear_solver_diverged = std::isnan (convergence.second);

            ++nonlin_iter;
            ++n_total_iter;

            // Out put convergence history
            iteration_history_file
                << std::setw (7) << n_total_iter << ' '
                << std::setw (10) << triangulation.n_global_active_cells() << ' '
                << std::setw (10) << dof_handler.n_dofs() << ' '
                << std::setw (13) << time << ' '
                << std::setw (8) << n_time_step << ' '
                << std::setw (9) << nonlin_iter << ' '
                << std::setw (13) << res_norm << ' '
                << std::setw (14) << convergence.first << ' '
                << std::setw (13) << convergence.second << ' '
                << std::setw (18) << linear_search_length[index_linear_search_length] << ' '
                << std::setw (15) << global_time_step_size << ' '
                << std::setw (13) << CFL_number << ' '
                << std::setw (19) << newton_update_norm
                << '\n';
            // Check result.
            newton_iter_converged = (res_norm < terminal_res);

            if (linear_solver_diverged)
              {
                pcout << "  Linear solver diverged..\n";
              }
            if (solution_diverged)
              {
                pcout << "  Solution diverged..\n";
              }
            // May 'newton_iter_converged' and 'linear_solver_diverged' be true
            // together? I don't think so but not sure.


            // If linear solver diverged or Newton iteration not converge in
            // a reasonable number of iterations, we terminate this time step and set
            // 'newton_iter_converged' to false. Then further action will be taken
            // to handle this situation.

            //                                 Using '>='  here because this condition
            //                                 is evaluated after '++nonlin_iter'.
            if (linear_solver_diverged
                || solution_diverged
                || nonlin_iter >= nonlin_iter_threshold)
              {
                newton_iter_converged = false;
                pcout << "  Newton iteration not converge in " << nonlin_iter_threshold << " steps.\n";
              }
            computing_timer.leave_subsection ("5:Postprocess Newton solution");
          }
        while ((!solution_diverged)
               && (!newton_iter_converged)
               && nonlin_iter < nonlin_iter_threshold
               && (!linear_solver_diverged));

        if (newton_iter_converged)
          {
            computing_timer.enter_subsection ("6:Postprocess time step");
            is_refine_step = !is_refine_step;

            WallForce wall_force;
            integrate_force (wall_force);


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
                << std::setw (7) << n_total_iter << ' '
                << std::setw (10) << triangulation.n_global_active_cells() << ' '
                << std::setw (10) << dof_handler.n_dofs() << ' '
                << std::setw (13) << time << ' '
                << std::setw (8) << n_time_step << ' '
                << std::setw (9) << nonlin_iter << ' '
                << std::setw (13) << res_norm << ' '
                << std::setw (14) << convergence.first << ' '
                << std::setw (13) << convergence.second << ' '
                << std::setw (18) << linear_search_length[index_linear_search_length] << ' '
                << std::setw (15) << global_time_step_size << ' '
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
            // old_solution.  This simple time extrapolation does the job. With
            // this, we then refine the mesh if so desired by the user, and
            // finally continue on with the next time step:
            ++converged_newton_iters;
            if (!parameters->is_steady)
              {
                time += global_time_step_size;
              }
            ++n_time_step;
            terminate_time_stepping = terminate_time_stepping || time >= parameters->final_time;
            terminate_time_stepping = terminate_time_stepping ||
                                      (parameters->is_steady &&
                                       n_time_step > parameters->final_time);
            bool do_refine = (
                               (parameters->is_steady)
                               ?
                               (static_cast<unsigned int> (parameters->max_refine_time) >= n_time_step)
                               :
                               (parameters->max_refine_time >= time)
                             );
            do_refine = do_refine &&
                        (! (parameters->dodge_mesh_adaptation) || is_refine_step);
            if (parameters->laplacian_continuation > 0.0)
              {
                do_refine = do_refine &&
                            n_time_step <= n_step_laplacian_vanished + parameters->max_refine_level;
              }

            if (do_refine)
              {
                compute_refinement_indicators();
              }

            if (parameters->is_steady)
              {
                if (n_time_step >= next_output_time_step)
                  {
                    output_results();
                    next_output_time_step += int_output_step;
                  }
              }
            else
              {
                if (time >= next_output)
                  {
                    output_results();
                    next_output += parameters->output_step;
                  }
              }

            if (parameters-> is_steady)
              {
                if (! (parameters->dodge_mesh_adaptation) || is_refine_step)
                  {
                    if (n_time_step <= parameters-> n_iter_stage1)
                      {
                        CFL_number *= parameters->step_increasing_ratio_stage1;
                      }
                    else
                      {
                        double const
                        ratio = std::max (parameters->minimum_step_increasing_ratio_stage2,
                                          std::pow (res_norm_total_previous/res_norm_total, parameters->step_increasing_power_stage2));
                        CFL_number *= ratio;
                      }
                    CFL_number = std::min (CFL_number, parameters->CFL_number_max);
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
            const bool continuation_coefficient_vanished =
              (continuation_coefficient < 1e-100);

            // TODO: refactoring needed.
            // Never decease continuation_coefficient and do mesh refine together.
            if (! (parameters->dodge_mesh_adaptation) || !is_refine_step)
              {
                old_continuation_coefficient = continuation_coefficient;
                {
                  n_step_laplacian_vanished += (continuation_coefficient>0.0);
                  double laplacian_ratio_min = 0.5;
                  if (quadratic_converge && parameters->Mach < 0.5)
                    {
                      laplacian_ratio_min =
                        std::min (0.1,
                                  std::pow (continuation_coefficient,
                                            parameters->laplacian_decrease_rate - 1.0)
                                 );
                    }
                  const double laplacian_ratio =
                    std::min (laplacian_ratio_min,
                              0.9 * physical_residual_ratio);

                  continuation_coefficient *= laplacian_ratio;
                  if (continuation_coefficient < parameters->laplacian_zero * mean_artificial_viscosity)
                    {
                      if (n_step_laplacian_vanished > 65500)
                        {
                          n_step_laplacian_vanished = n_time_step;
                        }
                      continuation_coefficient = 0.0;
                    }
                }
              }

            pcout << "res_norm_total = " << res_norm_total << std::endl;
            pcout << "continuation_coefficient = " << continuation_coefficient << std::endl;

            std_cxx11::array<double, EquationComponents<dim>::n_components> time_advance_l2_norm;
            for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
              {
                mms_error_l2[ic] = 0.0;
                mms_error_H1_semi[ic] = 0.0;
                time_advance_l2_norm[ic] = 0.0;
              }
            mms_error_linfty = 0.0;

            std::vector<Vector<double> > solution_values;
            std::vector< std::vector< Tensor<1, dim> > > solution_grad;
            std::vector<Vector<double> > old_solution_values;
            //MMS: update quadrature points for evaluation of manufactured solution.
            const UpdateFlags update_flags = update_values
                                             | update_gradients
                                             | update_JxW_values
                                             | update_quadrature_points;
            const QGauss<dim> quadrature_error (parameters->error_quadrature_degree);
            FEValues<dim>  fe_v (*mapping_ptr, fe, quadrature_error, update_flags);

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
                  solution_grad.resize (n_q_points,
                                        std::vector< Tensor<1, dim> > (EquationComponents<dim>::n_components));
                  old_solution_values.resize (n_q_points,
                                              Vector<double> (EquationComponents<dim>::n_components));
                  fe_v.get_function_values (current_solution, solution_values);
                  fe_v.get_function_gradients (current_solution, solution_grad);
                  fe_v.get_function_values (old_solution, old_solution_values);

                  std::vector <typename MMS<dim>::F_V> mms_source (n_q_points);
                  std::vector <typename MMS<dim>::F_V> mms_value (n_q_points);
                  std::vector <typename MMS<dim>::F_T> mms_grad (n_q_points);

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
                          mms.evaluate (fe_v.quadrature_point (q), mms_value[q], mms_grad[q],
                                        mms_source[q], /* const bool need_source = */ false);
                        }

                      for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
                        {
                          for (unsigned int q=0; q<n_q_points; ++q)
                            {
                              mms_error_l2[ic] += (mms_value[q][ic] - solution_values[q][ic]) *
                                                  (mms_value[q][ic] - solution_values[q][ic]) *
                                                  fe_v.JxW (q);
                              double grad_diff = 0.0;
                              for (unsigned int d=0; d<dim; ++d)
                                {
                                  grad_diff += (mms_grad[q][ic][d] - solution_grad[q][ic][d]) *
                                               (mms_grad[q][ic][d] - solution_grad[q][ic][d]);
                                }
                              mms_error_H1_semi[ic] += grad_diff * fe_v.JxW (q);
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
                    const double received_sum =
                      Utilities::MPI::sum (mms_error_l2[ic], mpi_communicator);
                    mms_error_l2[ic] = received_sum;
                    pcout << 0.5 * std::log10 (mms_error_l2[ic]) << ' ';
                  }
                pcout << std::endl;
                pcout << "  MMS Error H1-semi norm:\n";
                pcout << "    n_dofs    u_err    v_err  rho_err    p_err (log10)\n   ";
                pcout <<  std::log10 (dof_handler.n_dofs()) << ' ';
                for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
                  {
                    const double received_sum =
                      Utilities::MPI::sum (mms_error_H1_semi[ic], mpi_communicator);
                    mms_error_H1_semi[ic] = received_sum;
                    pcout << 0.5 * std::log10 (mms_error_H1_semi[ic]) << ' ';
                  }
                pcout << std::endl;
              }

            // Only try to switch flux type when current and target flux
            // types are different.
            bool swith_flux =
              parameters->numerical_flux_type != parameters->flux_type_switch_to;
            // If switch of flux type is requested and the flux switch tolerance is
            // larger than time march tolerance, never stop time marching
            // before switching to the target flux type.
            bool time_march_converged =
              ! (swith_flux &&
                 parameters->tolerance_to_switch_flux > parameters->time_march_tolerance);
            time_march_converged = time_march_converged &&
                                   continuation_coefficient_vanished;

            pcout << "  Order of time advancing L_2  norm\n   ";
            double total_time_march_norm = 0.0;
            for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
              {
                double receive_sum;
                receive_sum = Utilities::MPI::sum (time_advance_l2_norm[ic], mpi_communicator);
                time_advance_l2_norm[ic] = receive_sum;
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

            if (parameters->laplacian_continuation > 0.0)
              {
                time_march_converged =
                  (n_time_step > n_step_laplacian_vanished + parameters->max_refine_level + 2)
                  &&
                  (res_norm < 1e-11);
              }

            if (swith_flux)
              {
                parameters_modifier->numerical_flux_type = parameters->flux_type_switch_to;
              }
            terminate_time_stepping = terminate_time_stepping ||
                                      (parameters->is_steady &&
                                       time_march_converged);

            old_old_solution = old_solution;
            old_solution = current_solution;
            if (do_refine)
              {
                refine_grid();
              }
            old_artificial_viscosity = artificial_viscosity;
            old_time_step_size = global_time_step_size;
            check_negative_density_pressure();
            calc_time_step();
            // Uncomment the following line if you want reset the linear_search_length immediately after a converged Newton iter.
            //index_linear_search_length = 0;
            computing_timer.leave_subsection ("6:Postprocess time step");
          }
        else
          {
            computing_timer.enter_subsection ("6:Rolling back time step");
            // Newton iteration not converge in reasonable steps
            if (parameters->laplacian_continuation > 0.0)
              {
                if (continuation_coefficient > 0.0)
                  {
                    continuation_coefficient =
                      std::sqrt (old_continuation_coefficient * continuation_coefficient);
                  }
                else
                  {
                    continuation_coefficient = 0.5 * old_continuation_coefficient;
                  }
                pcout << "reseted continuation_coefficient = " << continuation_coefficient << std::endl;
              }
            else if ((index_linear_search_length <
                      parameters->newton_linear_search_length_try_limit)
                     && (!CFL_number_increased))
              {
                // Try to adjust linear_search_length first
                ++index_linear_search_length;
              }
            else
              {
                AssertThrow (parameters->allow_decrease_CFL,
                             ExcMessage ("Nonlinear not convergence and reducing CFL number is disabled."));
                // Reduce time step when linear_search_length has tried out.
                CFL_number *= 0.5;
                AssertThrow (CFL_number >= parameters->CFL_number_min,
                             ExcMessage ("No convergence in nonlinear solver after all small time step and linear search length tried out."));

                pcout << "  Recompute with different linear search length or time step...\n\n";
                global_time_step_size *= 0.5;
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
                                         global_time_step_size / old_time_step_size;
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

// Clean run time allocated HyperBallBoundary object.
    if (parameters->manifold_circle > 0)
      {
        triangulation.set_boundary (2, straight_boundary);
        delete spherical_boundary;
        spherical_boundary = 0;
      }
  }

#include "NSolver.inst"
}
