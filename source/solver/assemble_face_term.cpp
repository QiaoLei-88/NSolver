//
//  NSolver::assemble_face_term.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II turorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


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

#include "NSolver.inst"
}
