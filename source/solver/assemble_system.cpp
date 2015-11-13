//
//  NSolver::assemble_system.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


  // @sect4{NSolver::assemble_system}
  //
  // This and the following two functions are the meat of this program: They
  // assemble the linear system that results from applying Newton's method to
  // the nonlinear system of conservation equations.
  //
  // This first function puts all of the assembly pieces together in a routine
  // that dispatches the correct piece for each cell/face.  The actual
  // implementation of the assembly on these objects is done in the following
  // functions.
  //
  // At the top of the function we do the usual housekeeping: allocate
  // FEValues, FEFaceValues, and FESubfaceValues objects necessary to do the
  // integrations on cells, faces, and subfaces (in case of adjoining cells on
  // different refinement levels). Note that we don't need all information
  // (like values, gradients, or real locations of quadrature points) for all
  // of these objects, so we only let the FEValues classes whatever is
  // actually necessary by specifying the minimal set of UpdateFlags. For
  // example, when using a FEFaceValues object for the neighboring cell we
  // only need the shape values: Given a specific face, the quadrature points
  // and <code>JxW</code> values are the same as for the current cells, and
  // the normal vectors are known to be the negative of the normal vectors of
  // the current cell.
  template <int dim>
  void NSolver<dim>::assemble_system()
  {
    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> dof_indices (dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices_neighbor (dofs_per_cell);

    //MMS: update quadrature points for evaluation of manufactured solution.
    const UpdateFlags update_flags               = update_values
                                                   | update_gradients
                                                   | update_q_points
                                                   | update_JxW_values
                                                   | update_quadrature_points;
    const UpdateFlags face_update_flags          = update_values
                                                   | update_q_points
                                                   | update_JxW_values
                                                   | update_normal_vectors
                                                   | update_quadrature_points;
    const UpdateFlags neighbor_face_update_flags = update_values;

    FEValues<dim>        fe_v (*mapping_ptr, fe, quadrature,
                               update_flags);
    FEFaceValues<dim>    fe_v_face (*mapping_ptr, fe, face_quadrature,
                                    face_update_flags);
    FESubfaceValues<dim> fe_v_subface (*mapping_ptr, fe, face_quadrature,
                                       face_update_flags);
    FEFaceValues<dim>    fe_v_face_neighbor (*mapping_ptr, fe, face_quadrature,
                                             neighbor_face_update_flags);
    FESubfaceValues<dim> fe_v_subface_neighbor (*mapping_ptr, fe, face_quadrature,
                                                neighbor_face_update_flags);

    // Then loop over all cells, initialize the FEValues object for the current
    // cell and call the function that assembles the problem on this cell.
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_v.reinit (cell);
          cell->get_dof_indices (dof_indices);

          assemble_cell_term (fe_v, dof_indices);

          // Then loop over all the faces of this cell.  If a face is part of
          // the external boundary, then assemble boundary conditions there (the
          // fifth argument to <code>assemble_face_terms</code> indicates
          // whether we are working on an external or internal face; if it is an
          // external face, the fourth argument denoting the degrees of freedom
          // indices of the neighbor is ignored, so we pass an empty vector):
          for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
              const bool face_is_at_boundary = cell->at_boundary (face_no);
              FEFaceValuesBase<dim> *ptr_fe_v_face_this = 0;
              FEFaceValuesBase<dim> *ptr_fe_v_face_neighbor = 0;
              unsigned int neighbor_face_no = numbers::invalid_unsigned_int;
              unsigned int boundary_id = numbers::invalid_unsigned_int;
              double face_diameter = -1.0;
              bool need_to_compute_face_flux = true;

              // Loop through all sub components of current face and set proper parameters for
              // function assemble_face_term().
              // The following statement means, if the face has children, loop through all of
              // its children; if not, treat itself as its only child.
              const unsigned int n_subfaces =
                cell->face (face_no)->has_children() ?
                cell->face (face_no)->n_children() : 1;
              for (unsigned int subface_no=0; subface_no < n_subfaces; ++subface_no)
                {
                  if (face_is_at_boundary)
                    {
                      need_to_compute_face_flux = true;

                      fe_v_face.reinit (cell, face_no);
                      ptr_fe_v_face_this     = &fe_v_face;
                      ptr_fe_v_face_neighbor = ptr_fe_v_face_this;
                      std::fill (dof_indices_neighbor.begin(),
                                 dof_indices_neighbor.end(),
                                 numbers::invalid_unsigned_int);
                      neighbor_face_no = numbers::invalid_unsigned_int;
                      boundary_id = cell->face (face_no)->boundary_id();
                      face_diameter = cell->face (face_no)->diameter();
                    }

                  // The alternative is that we are dealing with an internal face. There
                  // are two cases that we need to distinguish: that this is a normal
                  // face between two cells at the same refinement level, and that it is
                  // a face between two cells of the different refinement levels.
                  //
                  // In the first case, there is nothing we need to do: we are using a
                  // continuous finite element, and face terms do not appear in the
                  // bilinear form in this case. The second case usually does not lead
                  // to face terms either if we enforce hanging node constraints
                  // strongly (as in all previous tutorial programs so far whenever we
                  // used continuous finite elements -- this enforcement is done by the
                  // ConstraintMatrix class together with
                  // DoFTools::make_hanging_node_constraints). In the current program,
                  // however, we opt to enforce continuity weakly at faces between cells
                  // of different refinement level, for two reasons: (i) because we can,
                  // and more importantly (ii) because we would have to thread the
                  // automatic differentiation we use to compute the elements of the
                  // Newton matrix from the residual through the operations of the
                  // ConstraintMatrix class. This would be possible, but is not trivial,
                  // and so we choose this alternative approach.
                  //
                  // What needs to be decided is which side of an interface between two
                  // cells of different refinement level we are sitting on.
                  //
                  // Let's take the case where the neighbor is more refined first. We
                  // then have to loop over the children of the face of the current cell
                  // and integrate on each of them. We sprinkle a couple of assertions
                  // into the code to ensure that our reasoning trying to figure out
                  // which of the neighbor's children's faces coincides with a given
                  // subface of the current cell's faces is correct -- a bit of
                  // defensive programming never hurts.
                  //
                  // We then call the function that integrates over faces; since this is
                  // an internal face, the fifth argument is false, and the sixth one is
                  // ignored so we pass an invalid value again:
                  else
                    {
                      const typename DoFHandler<dim>::cell_iterator
                      neighbor = cell->neighbor (face_no);

                      // Make sure not subtracting two unsigned integers.
                      const int this_cell_level = cell->level();
                      const int neighbor_active_cell_level = neighbor->level() + neighbor->has_children();

                      switch (this_cell_level-neighbor_active_cell_level)
                        {
                        case -1:
                        {
                          need_to_compute_face_flux = true;
                          // This cell is one level coarser than neighbor. So we have several subfaces
                          // to deal with. The outer loop will take care of the walking.
                          neighbor_face_no = cell->neighbor_of_neighbor (face_no);

                          const typename DoFHandler<dim>::active_cell_iterator neighbor_child =
                            cell->neighbor_child_on_subface (face_no, subface_no);

                          // Make sure we are talking about the same subface.
                          Assert (neighbor_child->face (neighbor_face_no) ==
                                  cell->face (face_no)->child (subface_no),
                                  ExcInternalError());

                          fe_v_subface.reinit (cell, face_no, subface_no);
                          fe_v_face_neighbor.reinit (neighbor_child, neighbor_face_no);
                          ptr_fe_v_face_this     = &fe_v_subface;
                          ptr_fe_v_face_neighbor = &fe_v_face_neighbor;

                          neighbor_child->get_dof_indices (dof_indices_neighbor);

                          boundary_id = numbers::invalid_unsigned_int;
                          face_diameter = neighbor_child->face (neighbor_face_no)->diameter();
                          break;
                        }
                        case 0:
                        {
                          // This cell and neighbor cell are at same level
                          need_to_compute_face_flux = false;
                          break;
                        }
                        case 1:
                        {
                          need_to_compute_face_flux = true;
                          // This cell is one level finer than neighbor. So we are now
                          // on a subface of neighbor cell.
                          const std::pair<unsigned int, unsigned int>
                          faceno_subfaceno = cell->neighbor_of_coarser_neighbor (face_no);
                          neighbor_face_no = faceno_subfaceno.first;
                          const unsigned int &neighbor_subface_no = faceno_subfaceno.second;

                          // Make sure we are talking about the same subface.
                          Assert (neighbor->neighbor_child_on_subface (neighbor_face_no,
                                                                       neighbor_subface_no)
                                  == cell,
                                  ExcInternalError());

                          fe_v_face.reinit (cell, face_no);
                          fe_v_subface_neighbor.reinit (neighbor,
                                                        neighbor_face_no,
                                                        neighbor_subface_no);
                          ptr_fe_v_face_this     = &fe_v_face;
                          ptr_fe_v_face_neighbor = &fe_v_subface_neighbor;

                          neighbor->get_dof_indices (dof_indices_neighbor);

                          boundary_id = numbers::invalid_unsigned_int;
                          face_diameter = cell->face (face_no)->diameter();
                          break;
                        }
                        default:
                        {
                          Assert (false,
                                  ExcMessage ("Refinement level difference between face neighbor cells can't greater than 1."));
                          break;
                        }
                        } // End switch level difference
                    } // End if (face_is_at_boundary) ... else.
                  if (need_to_compute_face_flux)
                    {
                      assemble_face_term (face_no,
                                          neighbor_face_no,
                                          *ptr_fe_v_face_this,
                                          *ptr_fe_v_face_neighbor,
                                          dof_indices,
                                          dof_indices_neighbor,
                                          face_is_at_boundary,
                                          boundary_id,
                                          face_diameter);
                    }
                } // End for all subfaces
            } // End for all faces
        } // End for all active cells

    // After all this assembling, notify the Trilinos matrix object that the
    // matrix is done:
    system_matrix.compress (VectorOperation::add);
    right_hand_side.compress (VectorOperation::add);
    physical_residual.compress (VectorOperation::add);

    return;
  }

#include "NSolver.inst"
}
