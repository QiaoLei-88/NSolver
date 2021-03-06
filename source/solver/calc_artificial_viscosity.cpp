//
//  NSolver::calc_artificial_viscosity.cpp
//
//  Created by Lei Qiao on 15/9/2.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;
  template <int dim>
  void
  NSolver<dim>::calc_artificial_viscosity()
  {
    // Find out global minimum cell size
    double local_h_min(std::numeric_limits<double>::max());

    typename DoFHandler<dim>::active_cell_iterator cell =
      dof_handler.begin_active();
    const typename DoFHandler<dim>::active_cell_iterator endc =
      dof_handler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            local_h_min = std::min(cell->diameter(), local_h_min);
          }
      }

    const double global_h_min =
      Utilities::MPI::min(local_h_min, mpi_communicator);
    // This is to say local_h_min will never be used here after.
    (void)local_h_min;
    const double global_effective_h_max =
      global_h_min * parameters->h_ceiling_ratio;

    switch (parameters->diffusion_type)
      {
        case Parameters::AllParameters<dim>::diffu_entropy:
          {
            // Entropy viscosity
            FEValues<dim> fe_values(fe,
                                    quadrature,
                                    update_values | update_gradients |
                                      update_quadrature_points);

            const unsigned int          n_q_points    = quadrature.size();
            const unsigned int          dofs_per_cell = fe_values.dofs_per_cell;
            std::vector<Vector<double>> W(
              n_q_points,
              Vector<double>(EquationComponents<dim>::n_components));
            std::vector<std::vector<Tensor<1, dim>>> grad_W(
              n_q_points,
              std::vector<Tensor<1, dim>>(
                EquationComponents<dim>::n_components));
            std::vector<Vector<double>> W_old(
              n_q_points,
              Vector<double>(EquationComponents<dim>::n_components));

            std::vector<types::global_dof_index> global_indices_of_local_dofs(
              dofs_per_cell);

            double                                         Mach_max = 0.0;
            typename DoFHandler<dim>::active_cell_iterator cell =
              dof_handler.begin_active();
            const typename DoFHandler<dim>::active_cell_iterator endc =
              dof_handler.end();
            for (; cell != endc; ++cell)
              {
                if (cell->is_locally_owned())
                  {
                    fe_values.reinit(cell);
                    fe_values.get_function_values(current_solution, W);
                    fe_values.get_function_gradients(current_solution, grad_W);
                    fe_values.get_function_values(old_solution, W_old);
                    const double dt =
                      parameters->use_local_time_step_size ?
                        local_time_step_size[cell->active_cell_index()] :
                        global_time_step_size;

                    cell->get_dof_indices(global_indices_of_local_dofs);

                    double rho_max(-1.0), D_h_max(-1.0),
                      characteristic_speed_max(-1.0);

                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        // Here, we need to evaluate the derivatives of entropy
                        // flux respect to Euler equation independent variables
                        // $w$ rather than the unknown vector $W$. So we have to
                        // set up a new Sacado::Fad::DFad system.
                        std::array<Sacado::Fad::DFad<double>,
                                   EquationComponents<dim>::n_components>
                          w_for_entropy_flux;
                        for (unsigned int c = 0;
                             c < EquationComponents<dim>::n_components;
                             ++c)
                          {
                            w_for_entropy_flux[c] = W[q][c];
                            w_for_entropy_flux[c].diff(
                              c, EquationComponents<dim>::n_components);
                          }

                        const Sacado::Fad::DFad<double> entropy =
                          EulerEquations<dim>::template compute_entropy(
                            w_for_entropy_flux);
                        const double entroy_old =
                          EulerEquations<dim>::template compute_entropy(
                            W_old[q]);

                        double D_h1(0.0), D_h2(0.0);
                        if (parameters->entropy_with_solution_diff)
                          {
                            D_h1 = (entropy.val() - entroy_old) / dt;
                            D_h2 =
                              (W[q]
                                [EquationComponents<dim>::density_component] -
                               W_old[q][EquationComponents<
                                 dim>::density_component]) /
                              dt;
                          }

                        // sum up divergence
                        for (unsigned int d = 0; d < dim; d++)
                          {
                            const Sacado::Fad::DFad<double> entropy_flux =
                              entropy *
                              w_for_entropy_flux[EquationComponents<dim>::
                                                   first_velocity_component +
                                                 d];
                            for (unsigned int c = 0;
                                 c < EquationComponents<dim>::n_components;
                                 ++c)
                              {
                                D_h1 += entropy_flux.fastAccessDx(c) *
                                        grad_W[q][c][d];
                              }
                            D_h2 +=
                              grad_W[q][EquationComponents<
                                          dim>::first_velocity_component +
                                        d][d] *
                                W[q]
                                 [EquationComponents<dim>::density_component] +
                              W[q][EquationComponents<
                                     dim>::first_velocity_component +
                                   d] *
                                grad_W[q][EquationComponents<
                                  dim>::density_component][d];
                          }
                        D_h2 *=
                          entropy.val() /
                          W[q][EquationComponents<dim>::density_component];
                        if (std::abs(D_h1) >= D_h_max)
                          {
                            dominant_viscosity[cell->active_cell_index()] = 2.0;
                          }
                        D_h_max = std::max(D_h_max, std::abs(D_h1));
                        if (std::abs(D_h2) >= D_h_max)
                          {
                            dominant_viscosity[cell->active_cell_index()] = 3.0;
                          }
                        D_h_max = std::max(D_h_max, std::abs(D_h2));

                        rho_max = std::max(
                          rho_max,
                          W[q][EquationComponents<dim>::density_component]);

                        const double sound_speed =
                          EulerEquations<dim>::compute_sound_speed(W[q]);
                        const double velocity =
                          EulerEquations<dim>::compute_velocity_magnitude(W[q]);
                        characteristic_speed_max =
                          std::max(characteristic_speed_max,
                                   velocity + sound_speed);
                        Mach_max = std::max(Mach_max, velocity / sound_speed);
                      }

                    const double h =
                      std::min(cell->diameter(), global_effective_h_max);

                    const double entropy_visc =
                      parameters->entropy_visc_cE * h * h * D_h_max;
                    const double miu_max = parameters->entropy_visc_cLinear *
                                           h * characteristic_speed_max;

                    artificial_viscosity[cell->active_cell_index()] =
                      std::min(miu_max, entropy_visc);
                    if (miu_max < entropy_visc)
                      {
                        dominant_viscosity[cell->active_cell_index()] = 1.0;
                      }
                    if (parameters->entropy_with_rho_max)
                      {
                        artificial_viscosity[cell->active_cell_index()] *=
                          rho_max;
                      }
                  } // End if cell is locally owned
              }     // End for active cells

            // blend refinement indicators with previous time step
            const double old_mu_l2  = old_artificial_viscosity.l2_norm();
            const double this_mu_l2 = artificial_viscosity.l2_norm();
            pcout << "old_mu_l2 = " << old_mu_l2 << std::endl
                  << "this_mu_l2 = " << this_mu_l2 << std::endl
                  << std::endl;
            blend_artificial_viscosity =
              blend_artificial_viscosity || (this_mu_l2 < old_mu_l2);
            if (blend_artificial_viscosity && Mach_max > 0.95)
              {
                // Scaling and addition of vector, i.e. *this.sadd(s,a,V) =
                // s*(*this)+a*V.
                artificial_viscosity.sadd(0.5, 0.5, old_artificial_viscosity);
                artificial_thermal_conductivity = artificial_viscosity;
              }
            pcout << "l2_blended_mu = " << artificial_viscosity.l2_norm()
                  << std::endl
                  << std::endl;
            break;
          }
        case Parameters::AllParameters<dim>::diffu_entropy_DRB:
          {
            // Declare constants used in this method
            const unsigned int &icp =
              EquationComponents<dim>::pressure_component;
            const unsigned int &icr =
              EquationComponents<dim>::density_component;
            const unsigned int &iv0 =
              EquationComponents<dim>::first_velocity_component;

            double Mach_max(std::numeric_limits<double>::min());
            // Begin computing artificial viscosity.
            const UpdateFlags update_flags =
              update_values | update_gradients | update_JxW_values;

            const UpdateFlags face_update_flags =
              update_values | update_normal_vectors | update_gradients;
            const UpdateFlags neighbor_face_update_flags = update_gradients;

            FEValues<dim>     fe_v(*mapping_ptr, fe, quadrature, update_flags);
            FEFaceValues<dim> fe_v_face(*mapping_ptr,
                                        fe,
                                        face_quadrature,
                                        face_update_flags);
            FESubfaceValues<dim> fe_v_subface(*mapping_ptr,
                                              fe,
                                              face_quadrature,
                                              face_update_flags);
            FEFaceValues<dim>    fe_v_face_neighbor(*mapping_ptr,
                                                 fe,
                                                 face_quadrature,
                                                 neighbor_face_update_flags);
            FESubfaceValues<dim> fe_v_subface_neighbor(
              *mapping_ptr, fe, face_quadrature, neighbor_face_update_flags);

            Vector<double> entropy(triangulation.n_active_cells());
            double         entropy_max = std::numeric_limits<double>::min();
            double         entropy_min = std::numeric_limits<double>::max();
            typename DoFHandler<dim>::active_cell_iterator cell =
              dof_handler.begin_active();
            const typename DoFHandler<dim>::active_cell_iterator endc =
              dof_handler.end();
            for (; cell != endc; ++cell)
              {
                if (!(cell->is_locally_owned()))
                  {
                    continue;
                  }

                // The following three variables will be set in next block and
                // be used in next next block. So we have to declare them here.
                //
                // viscosity_seed is the value that eventually used for
                // evaluation of entropy viscosity. It value is the maximum
                // among entropy production on all cell quadrature points and
                // density and pressure gradient jump on face quadratures.
                double viscosity_seed = std::numeric_limits<double>::min();
                double first_order_viscosity = -1;
                double scale_factor          = -1;

                // h: effective cell size. In context of mesh adaptation and
                // external aerodynamics problem, there are always very large
                // cells. Square there diameter will disturb the distribution
                // of artificial viscosity. Mean while, numerical examples in
                // literatures usually carried out on uniform mesh, thus h is
                // equivalent to its global min.
                const double h =
                  std::min(cell->diameter(), global_effective_h_max);

                {
                  // Here we start evaluation of cell entropy production,
                  // first order viscosity as upper bound of artificial
                  // viscosity, and Mach number based scale factor.
                  const unsigned int          n_q_points = quadrature.size();
                  std::vector<Vector<double>> W(
                    n_q_points,
                    Vector<double>(EquationComponents<dim>::n_components));
                  std::vector<std::vector<Tensor<1, dim>>> grad_W(
                    n_q_points,
                    std::vector<Tensor<1, dim>>(
                      EquationComponents<dim>::n_components));
                  fe_v.reinit(cell);
                  fe_v.get_function_values(current_solution, W);
                  fe_v.get_function_gradients(current_solution, grad_W);

                  // Compute maximum entropy production among all cell volume
                  // quadrature points
                  double max_entropy_production =
                    std::numeric_limits<double>::min();

                  // Cell average of Mach number
                  double local_Mach = 0;

                  // max_characteristic_speed is used for evaluation of first
                  // order viscosity
                  double max_characteristic_speed =
                    std::numeric_limits<double>::min();
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                      const double sound_speed_suqare =
                        parameters->gas_gamma * W[q][icp] / W[q][icr];

                      double entropy_production = 0.0;
                      // uu means square of velocity magnitude.
                      double uu = 0.0;
                      for (unsigned int d = 0, vd = iv0; d < dim; ++d, ++vd)
                        {
                          entropy_production +=
                            W[q][vd] * (grad_W[q][icp][d] -
                                        sound_speed_suqare * grad_W[q][icr][d]);
                          uu += W[q][vd] * W[q][vd];
                        }
                      // TODO: check negative entropy production.
                      max_entropy_production =
                        std::max(max_entropy_production,
                                 std::abs(entropy_production));
                      const double velocity_magnitude = std::sqrt(uu);
                      const double sound_speed = std::sqrt(sound_speed_suqare);
                      local_Mach +=
                        velocity_magnitude / sound_speed * fe_v.JxW(q);

                      max_characteristic_speed =
                        std::max(max_characteristic_speed,
                                 velocity_magnitude + sound_speed);
                    }
                  viscosity_seed =
                    std::max(viscosity_seed, max_entropy_production);
                  if (max_entropy_production >= viscosity_seed)
                    {
                      dominant_viscosity[cell->active_cell_index()] = 1.0;
                    }
                  // compute scale factor
                  local_Mach /= cell->measure();
                  Mach_max = std::max(Mach_max, local_Mach);
                  (void)local_Mach;
                  const double Mach = parameters->Mach;
                  scale_factor      = 1.0 / (Mach * Mach);

                  // First order viscosity
                  first_order_viscosity = 0.5 * h * max_characteristic_speed;
                } // End entropy production, scale factor and first order
                  // viscosity.

                {
                  // Here we start evaluation of face gradient jump.
                  const unsigned int n_q_points = face_quadrature.size();
                  std::vector<Vector<double>> W(
                    n_q_points,
                    Vector<double>(EquationComponents<dim>::n_components));
                  std::vector<Vector<double>> W_neighbor(
                    n_q_points,
                    Vector<double>(EquationComponents<dim>::n_components));
                  std::vector<std::vector<Tensor<1, dim>>> grad_W(
                    n_q_points,
                    std::vector<Tensor<1, dim>>(
                      EquationComponents<dim>::n_components));
                  std::vector<std::vector<Tensor<1, dim>>> grad_W_neighbor(
                    n_q_points,
                    std::vector<Tensor<1, dim>>(
                      EquationComponents<dim>::n_components));

                  // Compute maximum density and pressure gradient jump among
                  // all cell face quadrature points.
                  double max_gradient_jump = std::numeric_limits<double>::min();
                  float  dominant_jump     = 1.5;
                  for (unsigned int face_no = 0;
                       face_no < GeometryInfo<dim>::faces_per_cell;
                       ++face_no)
                    {
                      // We assume zero gradient jump on boundary
                      if (cell->at_boundary(face_no))
                        {
                          continue;
                        }
                      // The neighboring cell may not at the same refine level
                      // as the current cell, we have to handle different cases
                      // properly. First, the simplest case, neighboring cell is
                      // at the same refine level as the current one.
                      const typename DoFHandler<dim>::cell_iterator
                                neighbor_cell   = cell->neighbor(face_no);
                      const int this_cell_level = cell->level();
                      const int neighbor_active_cell_level =
                        neighbor_cell->level() + neighbor_cell->has_children();
                      if (this_cell_level == neighbor_active_cell_level)
                        {
                          // In CG mode, penalize against gradient jump
                          const unsigned int face_no_neighbor =
                            cell->neighbor_of_neighbor(face_no);

                          fe_v_face.reinit(cell, face_no);
                          fe_v_face_neighbor.reinit(neighbor_cell,
                                                    face_no_neighbor);

                          fe_v_face.get_function_values(current_solution, W);
                          fe_v_face.get_function_gradients(current_solution,
                                                           grad_W);
                          fe_v_face_neighbor.get_function_gradients(
                            current_solution, grad_W_neighbor);

                          // Gradient jump evaluation.
                          for (unsigned int q = 0; q < n_q_points; ++q)
                            {
                              const double sound_speed_suqare =
                                parameters->gas_gamma * W[q][icp] / W[q][icr];

                              const Tensor<1, dim> normal_vector =
                                fe_v_face.normal_vector(q);
                              const double pressure_gradient_jump = std::abs(
                                normal_vector *
                                (grad_W[q][icp] - grad_W_neighbor[q][icp]));
                              const double density_gradient_jump = std::abs(
                                normal_vector *
                                (grad_W[q][icr] - grad_W_neighbor[q][icr]));
                              const double jump_this_q =
                                EulerEquations<dim>::compute_velocity_magnitude(
                                  W[q]) *
                                std::max(pressure_gradient_jump,
                                         sound_speed_suqare *
                                           density_gradient_jump);
                              if (pressure_gradient_jump <
                                  sound_speed_suqare * density_gradient_jump)
                                {
                                  dominant_jump = 2.5;
                                }

                              max_gradient_jump =
                                std::max(max_gradient_jump, jump_this_q);
                            }
                        }
                      else if (neighbor_active_cell_level > this_cell_level)
                        {
                          // In DG mode, penalize against solution jump
                          Assert(
                            neighbor_active_cell_level == this_cell_level + 1,
                            ExcMessage(
                              "Refine level difference can't larger than 1 across cell interface."));
                          // Neighbor cell is refiner than this cell, we have
                          // to loop through all subfaces of this face.
                          const unsigned int face_no_neighbor =
                            cell->neighbor_of_neighbor(face_no);

                          for (unsigned int subface_no = 0;
                               subface_no < cell->face(face_no)->n_children();
                               ++subface_no)
                            {
                              const typename DoFHandler<
                                dim>::active_cell_iterator neighbor_child =
                                cell->neighbor_child_on_subface(face_no,
                                                                subface_no);

                              Assert(neighbor_child->face(face_no_neighbor) ==
                                       cell->face(face_no)->child(subface_no),
                                     ExcInternalError());
                              Assert(neighbor_child->has_children() == false,
                                     ExcInternalError());

                              fe_v_subface.reinit(cell, face_no, subface_no);
                              fe_v_face_neighbor.reinit(neighbor_child,
                                                        face_no_neighbor);

                              fe_v_subface.get_function_values(current_solution,
                                                               W);
                              fe_v_face_neighbor.get_function_values(
                                current_solution, W_neighbor);

                              // Solution jump evaluation.
                              for (unsigned int q = 0; q < n_q_points; ++q)
                                {
                                  const double sound_speed_suqare =
                                    parameters->gas_gamma * W[q][icp] /
                                    W[q][icr];

                                  const double pressure_value_jump =
                                    std::abs(W[q][icp] - W_neighbor[q][icp]);
                                  const double density_value_jump =
                                    std::abs(W[q][icr] - W_neighbor[q][icr]);
                                  const double jump_this_q =
                                    EulerEquations<
                                      dim>::compute_velocity_magnitude(W[q]) *
                                    std::max(pressure_value_jump,
                                             sound_speed_suqare *
                                               density_value_jump);
                                  if (pressure_value_jump <
                                      sound_speed_suqare * density_value_jump)
                                    {
                                      dominant_jump = 2.5;
                                    }
                                  max_gradient_jump =
                                    std::max(max_gradient_jump, jump_this_q);
                                }
                            }
                        }
                      else // if (neighbor_cell->level() < cell->level())
                        {
                          // In DG mode, penalize against solution jump
                          Assert(
                            neighbor_active_cell_level + 1 == this_cell_level,
                            ExcMessage(
                              "Refine level difference can't larger than 1 across cell interface."));
                          // Here, the neighbor cell is coarser than current
                          // cell. So we are on a subface of neighbor cell.

                          const std::pair<unsigned int, unsigned int>
                            faceno_subfaceno =
                              cell->neighbor_of_coarser_neighbor(face_no);
                          const unsigned int &neighbor_face_no =
                            faceno_subfaceno.first;
                          const unsigned int &neighbor_subface_no =
                            faceno_subfaceno.second;

                          Assert(neighbor_cell->neighbor_child_on_subface(
                                   neighbor_face_no, neighbor_subface_no) ==
                                   cell,
                                 ExcInternalError());

                          fe_v_face.reinit(cell, face_no);
                          fe_v_subface_neighbor.reinit(neighbor_cell,
                                                       neighbor_face_no,
                                                       neighbor_subface_no);

                          fe_v_face.get_function_values(current_solution, W);
                          fe_v_subface_neighbor.get_function_values(
                            current_solution, W_neighbor);

                          // Solution jump evaluation.
                          for (unsigned int q = 0; q < n_q_points; ++q)
                            {
                              const double sound_speed_suqare =
                                parameters->gas_gamma * W[q][icp] / W[q][icr];

                              const double pressure_value_jump =
                                std::abs(W[q][icp] - W_neighbor[q][icp]);
                              const double density_value_jump =
                                std::abs(W[q][icr] - W_neighbor[q][icr]);
                              const double jump_this_q =
                                EulerEquations<dim>::compute_velocity_magnitude(
                                  W[q]) *
                                std::max(pressure_value_jump,
                                         sound_speed_suqare *
                                           density_value_jump);
                              if (pressure_value_jump <
                                  sound_speed_suqare * density_value_jump)
                                {
                                  dominant_jump = 2.5;
                                }
                              max_gradient_jump =
                                std::max(max_gradient_jump, jump_this_q);
                            }
                        }
                    } // End loop for all faces
                  viscosity_seed = std::max(viscosity_seed, max_gradient_jump);
                  if (max_gradient_jump >= viscosity_seed)
                    {
                      dominant_viscosity[cell->active_cell_index()] =
                        dominant_jump;
                    }
                } // End gradient jump block

                entropy[cell->active_cell_index()] = viscosity_seed;
                entropy_min = std::min(viscosity_seed, entropy_min);
                entropy_max = std::max(viscosity_seed, entropy_max);

                // With all building blocks at hand, finally evaluate the
                // artificial viscosity.
                Assert(scale_factor > 0.0,
                       ExcMessage("scale_factor is negative"));
                const double second_order_viscosity =
                  h * h * viscosity_seed * scale_factor;
                if (first_order_viscosity <= second_order_viscosity)
                  {
                    dominant_viscosity[cell->active_cell_index()] = 3.0;
                  }
                artificial_viscosity[cell->active_cell_index()] =
                  parameters->entropy_visc_cE *
                  std::min(first_order_viscosity, second_order_viscosity);
                const double second_order_thermal_conductivity =
                  h * h * viscosity_seed * scale_factor;
                artificial_thermal_conductivity[cell->active_cell_index()] =
                  std::min(first_order_viscosity,
                           second_order_thermal_conductivity);
              } // End loop for all cells

            // Investigate max, min and average of artificial_viscosity
            {
              double local_max_v = std::numeric_limits<double>::min();
              double local_min_v = std::numeric_limits<double>::max();
              double local_sum_v = 0.0;
              {
                typename DoFHandler<dim>::active_cell_iterator
                  cell = dof_handler.begin_active(),
                  endc = dof_handler.end();
                for (; cell != endc; ++cell)
                  if (cell->is_locally_owned())
                    {
                      const double &v =
                        artificial_viscosity[cell->active_cell_index()];
                      local_max_v = std::max(local_max_v, v);
                      local_min_v = std::min(local_min_v, v);
                      local_sum_v += v;
                    }
              }
              const double max_v =
                Utilities::MPI::max(local_max_v, mpi_communicator);
              const double min_v =
                Utilities::MPI::min(local_min_v, mpi_communicator);
              const double avg_v =
                Utilities::MPI::sum(local_sum_v, mpi_communicator) /
                static_cast<double>(triangulation.n_global_active_cells());
              const double viscosity_threshold = avg_v * 2.0 - min_v;
              pcout << "viscosity min = " << min_v << std::endl
                    << "viscosity max = " << max_v << std::endl
                    << "viscosity avg = " << avg_v << std::endl
                    << "viscosity threshold = " << viscosity_threshold
                    << std::endl;

              {
                unsigned int n_small_mu = 0;
                typename DoFHandler<dim>::active_cell_iterator
                  cell = dof_handler.begin_active(),
                  endc = dof_handler.end();
                for (; cell != endc; ++cell)
                  if (cell->is_locally_owned())
                    {
                      double &v =
                        artificial_viscosity[cell->active_cell_index()];
                      if (v <= viscosity_threshold)
                        {
                          ++n_small_mu;
                          v = viscosity_threshold;
                        }
                    }
                pcout << "n_small_mu = "
                      << Utilities::MPI::sum(n_small_mu, mpi_communicator)
                      << std::endl;
              }
            }

            // blend refinement indicators with previous time step
            const double old_mu_l2  = old_artificial_viscosity.l2_norm();
            const double this_mu_l2 = artificial_viscosity.l2_norm();
            pcout << "old_mu_l2 = " << old_mu_l2 << std::endl
                  << "this_mu_l2 = " << this_mu_l2 << std::endl
                  << std::endl;
            blend_artificial_viscosity =
              blend_artificial_viscosity || (this_mu_l2 < old_mu_l2);
            if (blend_artificial_viscosity && Mach_max > 0.95)
              {
                // Scaling and addition of vector, i.e. *this.sadd(s,a,V) =
                // s*(*this)+a*V.
                artificial_viscosity.sadd(0.5, 0.5, old_artificial_viscosity);
                artificial_thermal_conductivity = artificial_viscosity;
              }
            pcout << "l2_blended_mu = " << artificial_viscosity.l2_norm()
                  << std::endl
                  << std::endl;
            const double entropy_mean = entropy.mean_value();
            const double scale_Guermond =
              std::max(entropy_mean - entropy_min, entropy_max - entropy_mean);
            pcout << "scale_Guermond = " << scale_Guermond << std::endl
                  << std::endl;
            break;
          }
        case Parameters::AllParameters<dim>::diffu_oscillation:
          {
            FEValues<dim> fe_values(fe,
                                    quadrature,
                                    update_values | update_JxW_values);

            const unsigned int          n_q_points = quadrature.size();
            std::vector<Vector<double>> W(
              n_q_points,
              Vector<double>(EquationComponents<dim>::n_components));

            double                                         Mach_max = 0.0;
            typename DoFHandler<dim>::active_cell_iterator cell =
              dof_handler.begin_active();
            const typename DoFHandler<dim>::active_cell_iterator endc =
              dof_handler.end();
            for (; cell != endc; ++cell)
              {
                if (cell->is_locally_owned())
                  {
                    fe_values.reinit(cell);
                    fe_values.get_function_values(current_solution, W);

                    std::array<double, EquationComponents<dim>::n_components>
                      cell_average;
                    std::fill(cell_average.begin(), cell_average.end(), 0.0);
                    const double cell_measure = cell->measure();

                    double characteristic_speed_max = 0.0;
                    if (parameters->oscillation_with_characteristic_speed)
                      {
                        for (unsigned int q = 0; q < n_q_points; ++q)
                          {
                            const double sound_speed =
                              EulerEquations<dim>::compute_sound_speed(W[q]);
                            const double velocity =
                              EulerEquations<dim>::compute_velocity_magnitude(
                                W[q]);
                            characteristic_speed_max =
                              std::max(characteristic_speed_max,
                                       velocity + sound_speed);
                          }
                      }
                    else
                      {
                        characteristic_speed_max = 1.0;
                      }

                    for (unsigned int c = 0;
                         c < EquationComponents<dim>::n_components;
                         ++c)
                      {
                        for (unsigned int q = 0; q < n_q_points; ++q)
                          {
                            cell_average[c] += W[q][c] * fe_values.JxW(q);
                          }
                        cell_average[c] /= cell_measure;
                      }

                    double top_order_inner_product = 0.0;
                    double solution_inner_product  = 0.0;
                    for (unsigned int c = 0;
                         c < EquationComponents<dim>::n_components;
                         ++c)
                      {
                        for (unsigned int q = 0; q < n_q_points; ++q)
                          {
                            top_order_inner_product +=
                              (W[q][c] - cell_average[c]) *
                              (W[q][c] - cell_average[c]) * fe_values.JxW(q);
                            solution_inner_product +=
                              W[q][c] * W[q][c] * fe_values.JxW(q);
                          }
                      }
                    const double oscillation_indicator = std::log10(
                      top_order_inner_product / solution_inner_product);
                    const double &visc_ceiling =
                      parameters->oscillation_visc_ceiling;
                    const double &visc_ground =
                      parameters->oscillation_visc_ground;
                    const double h =
                      std::min(cell->diameter(), global_effective_h_max);
                    double &oscillation_visc =
                      artificial_viscosity[cell->active_cell_index()];
                    oscillation_visc = parameters->oscillation_visc_background;
                    if (oscillation_indicator >= visc_ceiling)
                      {
                        oscillation_visc +=
                          characteristic_speed_max *
                          parameters->oscillation_visc_coefficient * 0.5 * h;
                      }
                    else if (oscillation_indicator > visc_ground)
                      {
                        const double gap  = visc_ceiling - visc_ground;
                        const double mean = 0.5 * (visc_ceiling + visc_ground);
                        oscillation_visc +=
                          characteristic_speed_max *
                          parameters->oscillation_visc_coefficient * 0.25 * h *
                          (1.0 +
                           std::sin(numbers::PI *
                                    (oscillation_indicator - mean) / gap));
                      }
                    dominant_viscosity[cell->active_cell_index()] =
                      oscillation_indicator;
                  } // End if cell is locally owned
              }     // End for active cells

            // blend refinement indicators with previous time step
            const double old_mu_l2  = old_artificial_viscosity.l2_norm();
            const double this_mu_l2 = artificial_viscosity.l2_norm();
            pcout << "old_mu_l2 = " << old_mu_l2 << std::endl
                  << "this_mu_l2 = " << this_mu_l2 << std::endl
                  << std::endl;
            blend_artificial_viscosity =
              blend_artificial_viscosity || (this_mu_l2 < old_mu_l2);
            if (blend_artificial_viscosity && Mach_max > -1.0)
              {
                // Scaling and addition of vector, i.e. *this.sadd(s,a,V) =
                // s*(*this)+a*V.
                artificial_viscosity.sadd(0.5, 0.5, old_artificial_viscosity);
                artificial_thermal_conductivity = artificial_viscosity;
              }
            pcout << "l2_blended_mu = " << artificial_viscosity.l2_norm()
                  << std::endl
                  << std::endl;
            break;
          }
        case Parameters::AllParameters<dim>::diffu_cell_size:
          {
            typename DoFHandler<dim>::active_cell_iterator cell =
              dof_handler.begin_active();
            const typename DoFHandler<dim>::active_cell_iterator endc =
              dof_handler.end();
            for (; cell != endc; ++cell)
              {
                if (cell->is_locally_owned())
                  {
                    artificial_viscosity[cell->active_cell_index()] =
                      parameters->diffusion_coefficoent *
                      std::pow(cell->diameter(), parameters->diffusion_power);
                  } // End if cell is locally owned
              }     // End for active cells

            // Next, compute global mean value of artificial viscosity.
            double local_sum_v = 0.0;
            {
              typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
              for (; cell != endc; ++cell)
                if (cell->is_locally_owned())
                  {
                    local_sum_v +=
                      artificial_viscosity[cell->active_cell_index()];
                  }
            }
            mean_artificial_viscosity =
              Utilities::MPI::sum(local_sum_v, mpi_communicator) /
              static_cast<double>(triangulation.n_global_active_cells());
            break;
          }
        case Parameters::AllParameters<dim>::diffu_const:
          {
            std::fill(artificial_viscosity.begin(),
                      artificial_viscosity.end(),
                      parameters->diffusion_coefficoent);
            mean_artificial_viscosity = parameters->diffusion_coefficoent;
            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
            break;
          }
      } // End switch case

    if (parameters->smooth_artificial_viscosity &&
        parameters->diffusion_type !=
          Parameters::AllParameters<dim>::diffu_const)
      {
        unsmoothed_artificial_viscosity.swap(artificial_viscosity);

        typename DoFHandler<dim>::active_cell_iterator cell =
          dof_handler.begin_active();
        const typename DoFHandler<dim>::active_cell_iterator endc =
          dof_handler.end();
        for (; cell != endc; ++cell)
          {
            if (!cell->is_locally_owned())
              {
                continue;
              }
            double max_visc =
              unsmoothed_artificial_viscosity[cell->active_cell_index()];
            // Loop through all neighbors of cell, and find out maximum value
            // of artificial viscosity;
            for (unsigned int face_no = 0;
                 face_no < GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
              {
                if (cell->at_boundary(face_no))
                  {
                    continue;
                  }

                const typename DoFHandler<dim>::cell_iterator &neighbor =
                  cell->neighbor(face_no);
                if (neighbor->is_active())
                  {
                    // Neighbor cell is on same level or coarser
                    if (neighbor->is_locally_owned())
                      {
                        max_visc = std::max(max_visc,
                                            unsmoothed_artificial_viscosity
                                              [neighbor->active_cell_index()]);
                      }
                  }
                else
                  {
                    // Neighbor cell is finer, need to loop through all subfaces
                    const unsigned int n_subfaces =
                      cell->face(face_no)->n_children();
                    for (unsigned int subface_no = 0; subface_no < n_subfaces;
                         ++subface_no)
                      {
                        const typename DoFHandler<dim>::active_cell_iterator
                          &neighbor_child =
                            cell->neighbor_child_on_subface(face_no,
                                                            subface_no);

                        // Make sure we find the right neighbor cell
                        Assert(neighbor_child->face(
                                 cell->neighbor_of_neighbor(face_no)) ==
                                 cell->face(face_no)->child(subface_no),
                               ExcInternalError());
                        if (neighbor_child->is_locally_owned() &&
                            neighbor_child->is_active())
                          {
                            max_visc =
                              std::max(max_visc,
                                       unsmoothed_artificial_viscosity
                                         [neighbor_child->active_cell_index()]);
                          }
                      }
                  } // if (neighbor->is_active())
              }     // for all faces

            artificial_viscosity[cell->active_cell_index()] = max_visc;
          } // for all cells
      }     // if (need do smooth)

    // Next, compute global mean value of artificial viscosity.
    double local_sum_v = 0.0;
    {
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                     endc = dof_handler.end();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            local_sum_v += artificial_viscosity[cell->active_cell_index()];
          }
    }
    mean_artificial_viscosity =
      Utilities::MPI::sum(local_sum_v, mpi_communicator) /
      static_cast<double>(triangulation.n_global_active_cells());

    pcout << "mean artificial_viscosity = " << mean_artificial_viscosity
          << std::endl
          << std::endl;

    return;
  } // End function

#include "NSolver.inst"
} // namespace NSFEMSolver
