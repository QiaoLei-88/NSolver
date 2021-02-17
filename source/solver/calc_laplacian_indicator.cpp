//
//  NSolver::calc_laplacian_indicator.cpp
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
  NSolver<dim>::calc_laplacian_indicator()
  {
    // First, compute laplacian_indicator vector.
    const UpdateFlags face_update_flags = update_values | update_gradients |
                                          update_normal_vectors |
                                          update_quadrature_points;
    const UpdateFlags neighbor_face_update_flags =
      update_values | update_gradients;

    FEFaceValues<dim>    fe_v_face(*mapping_ptr,
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
    FESubfaceValues<dim> fe_v_subface_neighbor(*mapping_ptr,
                                               fe,
                                               face_quadrature,
                                               neighbor_face_update_flags);

    // Then loop over all cells, initialize the FEValues object for the current
    // cell and call the function that assembles the problem on this cell.
    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              const bool face_is_at_boundary = cell->at_boundary(face_no);
              FEFaceValuesBase<dim> *ptr_fe_v_face_this     = 0;
              FEFaceValuesBase<dim> *ptr_fe_v_face_neighbor = 0;
              unsigned int boundary_id          = numbers::invalid_unsigned_int;
              bool         accumulate_grad_jump = false;
              bool         accumulate_value_jump = false;

              // Loop through all sub components of current face and set proper
              // parameters for function assemble_face_term(). The following
              // statement means, if the face has children, loop through all of
              // its children; if not, treat itself as its only child.
              const unsigned int n_subfaces =
                cell->face(face_no)->has_children() ?
                  cell->face(face_no)->n_children() :
                  1;
              for (unsigned int subface_no = 0; subface_no < n_subfaces;
                   ++subface_no)
                {
                  if (face_is_at_boundary)
                    {
                      accumulate_grad_jump  = false;
                      accumulate_value_jump = true;
                      fe_v_face.reinit(cell, face_no);
                      ptr_fe_v_face_this     = &fe_v_face;
                      ptr_fe_v_face_neighbor = ptr_fe_v_face_this;
                      boundary_id = cell->face(face_no)->boundary_id();
                    }
                  else
                    // The alternative is that we are dealing with an internal
                    // face.
                    {
                      const typename DoFHandler<dim>::cell_iterator neighbor =
                        cell->neighbor(face_no);

                      // Make sure not subtracting two unsigned integers.
                      const int this_cell_level = cell->level();
                      const int neighbor_active_cell_level =
                        neighbor->level() + neighbor->has_children();
                      switch (this_cell_level - neighbor_active_cell_level)
                        {
                          case -1:
                            {
                              accumulate_grad_jump  = true;
                              accumulate_value_jump = true;
                              // This cell is one level coarser than neighbor.
                              // So we have several subfaces to deal with. The
                              // outer loop will take care of the walking.
                              const unsigned int neighbor_face_no =
                                cell->neighbor_of_neighbor(face_no);

                              const typename DoFHandler<
                                dim>::active_cell_iterator neighbor_child =
                                cell->neighbor_child_on_subface(face_no,
                                                                subface_no);

                              // Make sure we are talking about the same
                              // subface.
                              Assert(neighbor_child->face(neighbor_face_no) ==
                                       cell->face(face_no)->child(subface_no),
                                     ExcInternalError());

                              fe_v_subface.reinit(cell, face_no, subface_no);
                              fe_v_face_neighbor.reinit(neighbor_child,
                                                        neighbor_face_no);
                              ptr_fe_v_face_this     = &fe_v_subface;
                              ptr_fe_v_face_neighbor = &fe_v_face_neighbor;
                              boundary_id = numbers::invalid_unsigned_int;

                              break;
                            }
                          case 0:
                            {
                              // This cell and neighbor cell are at same level
                              accumulate_grad_jump  = true;
                              accumulate_value_jump = false;

                              const unsigned int face_no_neighbor =
                                cell->neighbor_of_neighbor(face_no);

                              fe_v_face.reinit(cell, face_no);
                              fe_v_face_neighbor.reinit(neighbor,
                                                        face_no_neighbor);
                              ptr_fe_v_face_this     = &fe_v_face;
                              ptr_fe_v_face_neighbor = &fe_v_face_neighbor;

                              break;
                            }
                          case 1:
                            {
                              accumulate_grad_jump  = true;
                              accumulate_value_jump = true;
                              // This cell is one level finer than neighbor. So
                              // we are now on a subface of neighbor cell.
                              const std::pair<unsigned int, unsigned int>
                                faceno_subfaceno =
                                  cell->neighbor_of_coarser_neighbor(face_no);
                              const unsigned int &neighbor_face_no =
                                faceno_subfaceno.first;
                              const unsigned int &neighbor_subface_no =
                                faceno_subfaceno.second;

                              // Make sure we are talking about the same
                              // subface.
                              Assert(neighbor->neighbor_child_on_subface(
                                       neighbor_face_no, neighbor_subface_no) ==
                                       cell,
                                     ExcInternalError());

                              fe_v_face.reinit(cell, face_no);
                              fe_v_subface_neighbor.reinit(neighbor,
                                                           neighbor_face_no,
                                                           neighbor_subface_no);
                              ptr_fe_v_face_this     = &fe_v_face;
                              ptr_fe_v_face_neighbor = &fe_v_subface_neighbor;

                              boundary_id = numbers::invalid_unsigned_int;

                              break;
                            }
                          default:
                            {
                              Assert(
                                false,
                                ExcMessage(
                                  "Refinement level difference between face neighbor cells can't greater than 1."));
                              break;
                            }
                        } // End switch level difference
                    }     // End if (face_is_at_boundary) ... else.

                  laplacian_indicator[cell->active_cell_index()] +=
                    calc_jumps(*ptr_fe_v_face_this,
                               *ptr_fe_v_face_neighbor,
                               boundary_id,
                               accumulate_grad_jump,
                               accumulate_value_jump);
                } // End for all subfaces
            }     // End for all faces
        }         // End for all active cells

    // Next, compute laplacian indicator threshold.
    double local_max_v = std::numeric_limits<double>::min();
    double local_min_v = std::numeric_limits<double>::max();
    double local_sum_v = 0.0;
    {
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                     endc = dof_handler.end();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            const double &v = laplacian_indicator[cell->active_cell_index()];
            local_max_v     = std::max(local_max_v, v);
            local_min_v     = std::min(local_min_v, v);
            local_sum_v += v;
          }
    }
    const double max_v = Utilities::MPI::max(local_max_v, mpi_communicator);
    const double min_v = Utilities::MPI::min(local_min_v, mpi_communicator);
    const double avg_v =
      Utilities::MPI::sum(local_sum_v, mpi_communicator) /
      static_cast<double>(triangulation.n_global_active_cells());
    laplacian_threshold = avg_v * 2.0 - min_v;

    // Find threshold for a fixed cell number fraction;
    const bool laplacian_fix_number_fraction = true;
    if (laplacian_fix_number_fraction)
      {
        const double fraction =
          (1.0 * continuation_coefficient) /
          ((1.0 * continuation_coefficient) + mean_artificial_viscosity);

        const unsigned int n_target_cells =
          triangulation.n_global_active_cells() * fraction;
        const unsigned int master_mpi_rank      = 0;
        double             interesting_range[2] = {min_v, max_v};
        unsigned int       total_count          = 0;
        // Bisection search
        unsigned max_n_loop = 3;
        for (unsigned int error_range = triangulation.n_global_active_cells();
             error_range != 0;
             error_range /= 2)
          {
            ++max_n_loop;
          }
        for (unsigned int n = 0; n < max_n_loop; ++n)
          {
            laplacian_threshold =
              (interesting_range[0] + interesting_range[1]) * 0.5;


            // count cell number according to current test_threshold.
            unsigned int my_count = 0;
            {
              typename DoFHandler<dim>::active_cell_iterator cell =
                dof_handler.begin_active();
              const typename DoFHandler<dim>::active_cell_iterator endc =
                dof_handler.end();
              for (; cell != endc; ++cell)
                if (cell->is_locally_owned())
                  {
                    my_count +=
                      (laplacian_indicator[cell->active_cell_index()] >
                       laplacian_threshold);
                  }
            }

            MPI_Reduce(&my_count,
                       &total_count,
                       1,
                       MPI_UNSIGNED,
                       MPI_SUM,
                       master_mpi_rank,
                       mpi_communicator);
            if (total_count >= n_target_cells)
              {
                interesting_range[0] = laplacian_threshold;
              }
            if (total_count <= n_target_cells)
              {
                interesting_range[1] = laplacian_threshold;
              }
            MPI_Bcast(&interesting_range[0],
                      2,
                      MPI_DOUBLE,
                      master_mpi_rank,
                      mpi_communicator);
            if (interesting_range[1] == interesting_range[0])
              {
                break;
              }
          } // bisection iteration
      }     // End if (laplacian_fix_number_fraction)

    pcout << "laplacian min = " << min_v << std::endl
          << "laplacian max = " << max_v << std::endl
          << "laplacian avg = " << avg_v << std::endl
          << "laplacian threshold = " << laplacian_threshold << std::endl;

    if (parameters->refinement_indicator ==
        Parameters::Refinement<dim>::ErrorAndJump)
      {
        refinement_indicators = laplacian_indicator;
      }

    return;
  } // End function

#include "NSolver.inst"
} // namespace NSFEMSolver
